import os
import re
import unittest

HTML_FILE = 'snake_game.html'


class TestSnakeGameHTML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HTML_FILE)
        cls.file_exists = os.path.isfile(cls.file_path)
        cls.content = ''
        if cls.file_exists:
            with open(cls.file_path, 'r', encoding='utf-8') as f:
                cls.content = f.read()

    def test_file_exists(self):
        self.assertTrue(
            self.file_exists,
            f"Expected '{HTML_FILE}' to exist at {self.file_path}, but it was not found."
        )

    def test_file_non_empty(self):
        self.assertTrue(
            self.file_exists,
            f"Cannot check content: '{HTML_FILE}' does not exist."
        )
        self.assertGreater(
            len(self.content.strip()),
            0,
            f"'{HTML_FILE}' exists but is empty."
        )

    def test_canvas_element_present(self):
        self.assertRegex(
            self.content,
            r'<canvas[^>]*\bid=["\']gameCanvas["\']',
            "Missing <canvas id='gameCanvas'> element in the HTML file."
        )

    def test_difficulty_selector_present(self):
        # Look for a select element or similar construct offering difficulty options
        select_match = re.search(
            r'<select[^>]*>.*?</select>',
            self.content,
            re.IGNORECASE | re.DOTALL
        )
        self.assertIsNotNone(
            select_match,
            "No difficulty selector (<select> element) found in the HTML file."
        )

        selector_block = select_match.group(0)

        # Check for Easy, Medium, Hard options within the selector block
        for level in ('Easy', 'Medium', 'Hard'):
            self.assertRegex(
                selector_block,
                re.compile(re.escape(level), re.IGNORECASE),
                f"Difficulty selector is missing the '{level}' option."
            )

    def test_start_button_present(self):
        self.assertRegex(
            self.content,
            r'<button[^>]*\bid=["\']startBtn["\']',
            "Missing button with id 'startBtn' in the HTML file."
        )

    def test_score_element_present(self):
        self.assertRegex(
            self.content,
            r'id=["\']score["\']',
            "Missing element with id 'score' in the HTML file."
        )

    def test_script_tag_present(self):
        self.assertRegex(
            self.content,
            r'<script(?![^>]*\bsrc=)[^>]*>.*?</script>',
            re.IGNORECASE if False else 0
        ) if False else None
        script_match = re.search(
            r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>',
            self.content,
            re.IGNORECASE | re.DOTALL
        )
        self.assertIsNotNone(
            script_match,
            "No inline <script> block (without src attribute) found in the HTML file."
        )

    def _get_inline_script_content(self):
        matches = re.findall(
            r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>',
            self.content,
            re.IGNORECASE | re.DOTALL
        )
        self.assertTrue(
            len(matches) > 0,
            "No inline <script> block found to inspect game logic."
        )
        return "\n".join(matches)

    def test_script_contains_init_game_logic(self):
        script_content = self._get_inline_script_content()
        self.assertRegex(
            script_content,
            r'\binitGame\b',
            "Inline script does not reference 'initGame' function/logic."
        )

    def test_script_contains_game_loop_logic(self):
        script_content = self._get_inline_script_content()
        self.assertRegex(
            script_content,
            r'\bgameLoop\b',
            "Inline script does not reference 'gameLoop' function/logic."
        )

    def test_script_contains_difficulty_change_handler(self):
        script_content = self._get_inline_script_content()
        # Accept handleDifficultyChange or a reasonably equivalent named handler
        pattern = re.compile(
            r'\bhandleDifficultyChange\b'
            r'|\bonDifficultyChange\b'
            r'|\bchangeDifficulty\b'
            r'|\bsetDifficulty\b'
            r'|\bupdateDifficulty\b',
            re.IGNORECASE
        )
        self.assertRegex(
            script_content,
            pattern,
            "Inline script does not reference a difficulty-change handler "
            "(expected 'handleDifficultyChange' or an equivalent named function)."
        )

    def test_script_contains_speed_values(self):
        script_content = self._get_inline_script_content()
        for speed in ('200', '120', '70'):
            self.assertRegex(
                script_content,
                r'\b' + re.escape(speed) + r'\b',
                f"Inline script is missing the speed value '{speed}' "
                f"required for difficulty-based game speed."
            )

    def test_script_contains_collision_detection_logic(self):
        script_content = self._get_inline_script_content()

        # Wall / boundary collision detection keywords
        wall_pattern = re.compile(
            r'\bwall\b|\bboundary\b|\bboundaries\b|'
            r'<\s*0\b|>=\s*canvas\.width\b|>=\s*canvas\.height\b|'
            r'outOfBounds|checkWallCollision|checkBoundary',
            re.IGNORECASE
        )
        self.assertRegex(
            script_content,
            wall_pattern,
            "Inline script does not appear to contain wall/boundary collision detection logic."
        )

        # Self-collision detection keywords
        self_collision_pattern = re.compile(
            r'self[-_]?collision|checkSelfCollision|snake\.slice|'
            r'collideWithSelf|hitSelf',
            re.IGNORECASE
        )
        self.assertRegex(
            script_content,
            self_collision_pattern,
            "Inline script does not appear to contain self-collision detection logic."
        )

    def test_no_external_script_references(self):
        external_script_pattern = re.compile(
            r'<script[^>]+src=["\'][^"\']+["\']',
            re.IGNORECASE
        )
        matches = external_script_pattern.findall(self.content)
        self.assertEqual(
            len(matches),
            0,
            f"Found external <script src='...'> reference(s), but file must be self-contained: {matches}"
        )

    def test_no_external_stylesheet_references(self):
        external_css_pattern = re.compile(
            r'<link[^>]+rel=["\']stylesheet["\'][^>]*>',
            re.IGNORECASE
        )
        matches = external_css_pattern.findall(self.content)
        self.assertEqual(
            len(matches),
            0,
            f"Found external <link rel='stylesheet'> reference(s), but file must be self-contained: {matches}"
        )

    def test_no_cdn_url_references(self):
        cdn_pattern = re.compile(
            r'https?://[^\s"\'<>]*(cdn|jsdelivr|unpkg|googleapis|cloudflare)[^\s"\'<>]*',
            re.IGNORECASE
        )
        matches = cdn_pattern.findall(self.content)
        self.assertEqual(
            len(matches),
            0,
            f"Found CDN URL reference(s), but file must be fully self-contained: {matches}"
        )

    def test_no_generic_external_url_references(self):
        # Catch any remaining http(s):// URLs pointing to .js or .css files
        external_file_pattern = re.compile(
            r'https?://[^\s"\'<>]+\.(js|css)\b',
            re.IGNORECASE
        )
        matches = external_file_pattern.findall(self.content)
        self.assertEqual(
            len(matches),
            0,
            f"Found external .js/.css URL reference(s): {matches}"
        )


if __name__ == '__main__':
    unittest.main()