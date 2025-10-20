from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input
from textual.message import Message
from rich.text import Text

# The source text for the application
SOURCE_TEXT = (
    "In the heart of the bustling city, there stood a library of immense size and grandeur. "
    "Its shelves, carved from ancient oak, reached towards a ceiling painted with constellations. "
    "Every third book, it was rumored, contained not stories, but secrets. "
    "A young adventurer named Elara decided to test this legend. "
    "She walked past rows of dusty tomes, her fingers trailing along their spines. "
    "Finally, she found a small, leather-bound volume with no title. "
    "With a deep breath, she opened it, and the world around her faded away."
)


class StoryDisplay(Static):
    """A custom widget to display the story and handle clicks."""

    class BlankSelected(Message):
        """A message sent when a blank word is clicked."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def update_story(self, words: list[str], filled_blanks: dict[int, str]) -> None:
        """Re-renders the story using rich markup."""
        markup_parts = []
        for i, original_word in enumerate(words):
            is_blank_position = (i + 1) % 3 == 0

            if is_blank_position:
                click_action = f"select_blank({i})"
                if i in filled_blanks:
                    user_word = filled_blanks[i]
                    markup_parts.append(
                        f"[@click={click_action}][bold italic green]{user_word}[/][/]"
                    )
                else:
                    markup_parts.append(
                        f"[@click={click_action}][bold underline on blue]{original_word}[/][/]"
                    )
            else:
                markup_parts.append(original_word)

        final_markup = " ".join(markup_parts)
        renderable_text = Text.from_markup(final_markup)
        self.update(renderable_text)

    def action_select_blank(self, index: int) -> None:
        """Called when a user clicks a word. Posts a message to the app."""
        self.post_message(self.BlankSelected(index))


class MadLibsApp(App):
    """A Textual app for a fill-in-the-blanks story."""

    CSS_PATH = "styles.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.words = SOURCE_TEXT.split()
        self.active_blank_index = None
        self.filled_blanks = {}

    def compose(self) -> ComposeResult:
        """Create child widgets for the app in a logical vertical order."""
        yield Header()
        yield StoryDisplay(id="story_display")
        yield Input(
            placeholder="Click a blue word to begin...",
            id="word_input",
            classes="hidden",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted to the screen."""
        story_display = self.query_one(StoryDisplay)
        story_display.update_story(self.words, self.filled_blanks)
        story_display.focus()

    def on_story_display_blank_selected(
        self, message: StoryDisplay.BlankSelected
    ) -> None:
        """Handles the message from the StoryDisplay widget when a word is clicked."""
        index = message.index
        self.active_blank_index = index
        word_input = self.query_one("#word_input")

        word_input.remove_class("hidden")

        if index in self.filled_blanks:
            current_word = self.filled_blanks[index]
            word_input.value = current_word
            word_input.placeholder = f"Editing '{current_word}'..."
        else:
            original_word = self.words[index]
            word_input.value = original_word
            word_input.placeholder = f"Replacing '{original_word}'..."

        word_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when the user presses Enter in the Input widget."""
        word_input = event.control
        new_word = word_input.value.strip()
        story_display = self.query_one(StoryDisplay)

        if self.active_blank_index is not None and new_word:
            self.filled_blanks[self.active_blank_index] = new_word
            word_input.add_class("hidden")

            story_display.update_story(self.words, self.filled_blanks)

            self.active_blank_index = None
            word_input.value = ""
            word_input.placeholder = "Click a blue word above to continue..."
            story_display.focus()
        else:
            word_input.add_class("hidden")
            word_input.value = ""
            word_input.placeholder = "Click a blue word above to begin..."
            story_display.focus()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = MadLibsApp()
    app.run()
