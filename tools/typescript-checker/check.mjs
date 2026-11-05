import fs from "node:fs";
import path from "node:path";
import { builtinModules } from "node:module";
import ts from "typescript";

const MAX_FILES = 2000;
const MAX_SOURCE_CHARS = 4 * 1024 * 1024;
const MAX_DIAGNOSTICS = 200;
const MAX_IMPORTS = 5000;
const VIRTUAL_ROOT = "/pipeline-project";
const LIB_ROOT = path.resolve(path.dirname(ts.getDefaultLibFilePath({ target: ts.ScriptTarget.ES2022 })));
const sourceExtensions = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]);

function fail(message) {
  process.stdout.write(JSON.stringify({ status: "unsupported", limitations: [message] }));
  process.exit(0);
}

if (ts.version !== "6.0.3") {
  fail("The trusted TypeScript checker requires the pinned TypeScript 6.0.3 compiler.");
}

let request;
try {
  request = JSON.parse(fs.readFileSync(0, "utf8"));
} catch {
  fail("The trusted TypeScript checker received invalid bounded JSON input.");
}
if (!request || !Array.isArray(request.files) || request.files.length > MAX_FILES) {
  fail("TypeScript source count exceeds the checker input budget.");
}

const virtualFiles = new Map();
let sourceChars = 0;
for (const entry of request.files) {
  if (!entry || typeof entry.path !== "string" || typeof entry.content !== "string") {
    fail("A TypeScript checker source entry has an invalid shape.");
  }
  const normalized = entry.path.replaceAll("\\", "/");
  const absolute = path.resolve(VIRTUAL_ROOT, normalized);
  if (!absolute.startsWith(`${VIRTUAL_ROOT}${path.sep}`) || absolute === VIRTUAL_ROOT) {
    fail("A TypeScript checker source path escapes the authorized project snapshot.");
  }
  sourceChars += entry.content.length;
  if (sourceChars > MAX_SOURCE_CHARS) {
    fail("TypeScript source text exceeds the checker input budget.");
  }
  virtualFiles.set(absolute, entry.content);
}

const contracts = Array.isArray(request.contracts) ? request.contracts.slice(0, 20) : [];
const requiredContractNames = new Set(
  Array.isArray(request.requiredContractNames)
    ? request.requiredContractNames.filter((name) => typeof name === "string")
    : [],
);
const expectedNames = new Map();
const expectedDeclarations = [];
for (const [index, contract] of contracts.entries()) {
  if (!contract || typeof contract.name !== "string" || typeof contract.declaration !== "string") continue;
  const alias = `__PipelineExpected_${index}_${contract.name.replace(/[^A-Za-z0-9_$]/g, "_")}`;
  const declaration = contract.declaration.replace(
    /\b(interface|type)\s+[A-Za-z_$][A-Za-z0-9_$]*/,
    (_match, kind) => `${kind} ${alias}`,
  );
  if (declaration === contract.declaration) continue;
  expectedNames.set(contract.name, { alias, sourceRef: String(contract.source_ref ?? "planning input") });
  expectedDeclarations.push(declaration);
}
const contractFile = path.join(VIRTUAL_ROOT, "__pipeline_expected_contracts__.ts");
if (expectedDeclarations.length) {
  virtualFiles.set(contractFile, `export {};\n${expectedDeclarations.join("\n\n")}\n`);
}

const options = {
  allowJs: true,
  checkJs: false,
  declaration: false,
  incremental: false,
  composite: false,
  noEmit: true,
  noResolve: false,
  skipLibCheck: true,
  strict: true,
  target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
  jsx: ts.JsxEmit.Preserve,
  types: [],
  typeRoots: [],
  allowImportingTsExtensions: true,
};
const trustedLibRoot = fs.realpathSync(LIB_ROOT);
const canonical = (value) => path.resolve(value);
const within = (candidate, root) => candidate === root || candidate.startsWith(`${root}${path.sep}`);
const virtualDirectories = new Set([VIRTUAL_ROOT]);
for (const fileName of virtualFiles.keys()) {
  let current = path.dirname(fileName);
  while (within(current, VIRTUAL_ROOT)) {
    virtualDirectories.add(current);
    if (current === VIRTUAL_ROOT) break;
    current = path.dirname(current);
  }
}

function readAuthorized(fileName) {
  const absolute = canonical(fileName);
  if (virtualFiles.has(absolute)) return virtualFiles.get(absolute);
  if (!within(absolute, trustedLibRoot)) return undefined;
  try {
    const real = fs.realpathSync(absolute);
    if (!within(real, trustedLibRoot) || !fs.statSync(real).isFile()) return undefined;
    return fs.readFileSync(real, "utf8");
  } catch {
    return undefined;
  }
}

const host = ts.createCompilerHost(options, true);
host.getCurrentDirectory = () => VIRTUAL_ROOT;
host.getCanonicalFileName = (fileName) => fileName;
host.useCaseSensitiveFileNames = () => true;
host.getNewLine = () => "\n";
host.fileExists = (fileName) => readAuthorized(fileName) !== undefined;
host.readFile = readAuthorized;
host.directoryExists = (directoryName) => {
  const absolute = canonical(directoryName);
  return virtualDirectories.has(absolute) || within(absolute, trustedLibRoot);
};
host.getDirectories = (directoryName) => {
  const absolute = canonical(directoryName);
  return [...virtualDirectories]
    .filter((entry) => path.dirname(entry) === absolute)
    .map((entry) => path.basename(entry));
};
host.realpath = (fileName) => canonical(fileName);
host.getSourceFile = (fileName, languageVersion, onError) => {
  const content = readAuthorized(fileName);
  if (content === undefined) {
    onError?.(`Source path is outside the authorized TypeScript snapshot.`);
    return undefined;
  }
  const scriptKind = ts.getScriptKindFromFileName(fileName);
  return ts.createSourceFile(fileName, content, languageVersion, true, scriptKind);
};
host.writeFile = () => {};

const roots = [...virtualFiles.keys()];
let program;
try {
  program = ts.createProgram({ rootNames: roots, options, host });
} catch {
  fail("The pinned TypeScript compiler could not create a bounded source program.");
}

const sourceFiles = program.getSourceFiles().filter((source) => {
  const absolute = canonical(source.fileName);
  return virtualFiles.has(absolute) && absolute !== contractFile;
});
const modulesByFile = new Map();
const imports = [];
const builtins = new Set(builtinModules.flatMap((name) => [name, `node:${name}`]));
const contaminated = new Set();

function moduleReferences(source) {
  const references = [];
  function visit(node) {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      if (node.moduleSpecifier && ts.isStringLiteralLike(node.moduleSpecifier)) {
        references.push({ specifier: node.moduleSpecifier.text, kind: "static" });
      }
    } else if (ts.isCallExpression(node)) {
      const isImport = node.expression.kind === ts.SyntaxKind.ImportKeyword;
      const isRequire = ts.isIdentifier(node.expression) && node.expression.text === "require";
      if ((isImport || isRequire) && node.arguments.length === 1 && ts.isStringLiteralLike(node.arguments[0])) {
        references.push({ specifier: node.arguments[0].text, kind: isImport ? "dynamic" : "require" });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  return references;
}

for (const source of sourceFiles) {
  const fileName = canonical(source.fileName);
  const resolvedEdges = [];
  const references = moduleReferences(source);
  for (const reference of references) {
    if (imports.length >= MAX_IMPORTS) {
      fail("TypeScript import inventory exceeds the 5,000-import safety budget.");
    }
    const specifier = reference.specifier;
    let classification;
    let resolvedPath;
    if (specifier.startsWith("node:") || builtins.has(specifier)) {
      classification = "node_builtin";
      contaminated.add(fileName);
    } else {
      const resolution = ts.resolveModuleName(specifier, fileName, options, host).resolvedModule;
      if (resolution && virtualFiles.has(canonical(resolution.resolvedFileName))) {
        classification = specifier.startsWith(".") ? "local" : "project_alias";
        resolvedPath = path.relative(VIRTUAL_ROOT, canonical(resolution.resolvedFileName)).replaceAll(path.sep, "/");
        resolvedEdges.push(canonical(resolution.resolvedFileName));
      } else if (specifier.startsWith(".")) {
        classification = "local_unresolved";
      } else {
        classification = "external_unresolved";
        contaminated.add(fileName);
      }
    }
    imports.push({
      source: path.relative(VIRTUAL_ROOT, fileName).replaceAll(path.sep, "/"),
      specifier,
      kind: reference.kind,
      classification,
      ...(resolvedPath ? { resolved_path: resolvedPath } : {}),
    });
  }
  modulesByFile.set(fileName, resolvedEdges);
}

let expanded = true;
while (expanded) {
  expanded = false;
  for (const [fileName, dependencies] of modulesByFile) {
    if (!contaminated.has(fileName) && dependencies.some((dependency) => contaminated.has(dependency))) {
      contaminated.add(fileName);
      expanded = true;
    }
  }
}

const diagnostics = [];
function addDiagnostics(values, semantic = false) {
  for (const diagnostic of values) {
    if (diagnostic.file && semantic && contaminated.has(canonical(diagnostic.file.fileName))) continue;
    const fileName = diagnostic.file
      ? path.relative(VIRTUAL_ROOT, canonical(diagnostic.file.fileName)).replaceAll(path.sep, "/")
      : "";
    const position = diagnostic.file && typeof diagnostic.start === "number"
      ? diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start)
      : undefined;
    diagnostics.push({
      path: fileName,
      line: position ? position.line + 1 : null,
      code: diagnostic.code,
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, " ").slice(0, 800),
      kind: semantic ? "semantic" : "syntax",
    });
    if (diagnostics.length >= MAX_DIAGNOSTICS) return;
  }
}

addDiagnostics(program.getOptionsDiagnostics());
for (const source of sourceFiles) {
  if (diagnostics.length >= MAX_DIAGNOSTICS) break;
  addDiagnostics(program.getSyntacticDiagnostics(source));
}
for (const source of sourceFiles) {
  if (diagnostics.length >= MAX_DIAGNOSTICS) break;
  addDiagnostics(program.getSemanticDiagnostics(source), true);
}

const checker = program.getTypeChecker();
const contractFindings = [];
const limitations = [];
function namedTypeNode(name) {
  for (const source of sourceFiles) {
    let found;
    function visit(node) {
      if ((ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node)) && node.name.text === name) {
        found ??= node.name;
      }
      ts.forEachChild(node, visit);
    }
    visit(source);
    if (found) return found;
  }
  return undefined;
}
for (const [name, expected] of expectedNames) {
  const actualName = namedTypeNode(name);
  const expectedName = program.getSourceFile(contractFile)?.statements
    .flatMap((statement) => ts.isModuleDeclaration(statement) ? [] : [statement])
    .flatMap((statement) => (ts.isInterfaceDeclaration(statement) || ts.isTypeAliasDeclaration(statement)) ? [statement.name] : [])
    .find((identifier) => identifier.text === expected.alias);
  if (actualName && contaminated.has(canonical(actualName.getSourceFile().fileName))) {
    limitations.push(`Design contract ${name} from ${expected.sourceRef} was not compared because its owner depends on unresolved external or Node typings.`);
    continue;
  }
  if (!actualName || !expectedName) {
    if (request.requireAllContracts === true || requiredContractNames.has(name)) {
      contractFindings.push({ path: actualName?.getSourceFile().fileName ?? "", name, source_ref: expected.sourceRef, message: `Required named design contract ${name} was not declared in generated TypeScript sources.` });
    }
    continue;
  }
  const actualType = checker.getTypeAtLocation(actualName);
  const expectedType = checker.getTypeAtLocation(expectedName);
  if (!checker.isTypeAssignableTo(actualType, expectedType) || !checker.isTypeAssignableTo(expectedType, actualType)) {
    contractFindings.push({
      path: path.relative(VIRTUAL_ROOT, canonical(actualName.getSourceFile().fileName)).replaceAll(path.sep, "/"),
      name,
      source_ref: expected.sourceRef,
      message: `Generated ${name} is not structurally equivalent to the supplied contract source.`,
    });
  }
}

const unresolvedModules = [...new Set(imports
  .filter((entry) => entry.classification === "external_unresolved" || entry.classification === "node_builtin")
  .map((entry) => `${entry.specifier} (${entry.classification})`))].slice(0, 100);
if (unresolvedModules.length) {
  limitations.push(`TypeScript semantic diagnostics were suppressed for files contaminated by unresolved dependencies: ${unresolvedModules.join(", ")}.`);
}
if (sourceFiles.some((source) => [".js", ".jsx", ".mjs", ".cjs"].includes(path.extname(source.fileName).toLowerCase()))) {
  limitations.push("JavaScript/JSX syntax and imports were parsed; semantic type checking is enabled only for TypeScript sources (checkJs is disabled). ");
}

process.stdout.write(JSON.stringify({
  status: "checked",
  diagnostics,
  imports,
  contract_findings: contractFindings,
  limitations,
  checked_files: sourceFiles.length,
}));
