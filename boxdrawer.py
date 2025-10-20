import os
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Header,
    Footer,
    DirectoryTree,
    Static,
    Button,
    Input,
    Label,
)
from textual.message import Message


class ClickableBox(Static):
    """A clickable box widget that sends a message when clicked."""

    class Clicked(Message):
        """Message posted when a box is clicked, containing the box instance."""

        def __init__(self, box: "ClickableBox") -> None:
            self.box = box
            super().__init__()

    def on_click(self, event: events.Click) -> None:
        """Called when the user clicks the widget."""
        event.stop()
        self.post_message(self.Clicked(self))


class NewDirectoryScreen(ModalScreen[str]):
    """Screen with a dialog to enter a new directory name."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the modal dialog."""
        yield Vertical(
            Label("Create New Directory", id="dialog-title"),
            Label("Enter directory name:", classes="new-dir-label"),
            Input(placeholder="new-folder", id="new-dir-input"),
            Horizontal(
                Button("Create", variant="primary", id="create"),
                Button("Cancel", id="cancel"),
                classes="buttons",
            ),
            id="dialog",
        )

    def on_mount(self) -> None:
        """Focus the input when the screen is mounted."""
        self.query_one(Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses to create or cancel."""
        if event.button.id == "create":
            value = self.query_one(Input).value
            # Dismiss the screen, returning the value from the input.
            self.dismiss(value.strip())
        else:
            # Dismiss the screen, returning an empty string to indicate cancellation.
            self.dismiss("")


class DirectoryBoxApp(App[None]):
    """
    A Textual application with a directory tree and a clickable drawing area.
    """

    CSS = """
    Screen {
        overflow: hidden;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }
    
    #left-sidebar {
        width: 30;
        height: 100%;
        overflow: auto;
        border-right: thick $background-lighten-2;
        padding: 0 1;
        /* Use a vertical layout to stack the tree and the button */
        layout: vertical;
    }
    
    /* The DirectoryTree should take up all available space except for the button */
    #tree {
        height: 1fr;
        width: 100%;
    }
    
    #add-dir-button {
        /* Keep the button at the bottom of the sidebar */
        dock: bottom;
        width: 100%;
        margin-top: 1;
    }

    #main-area {
        width: 1fr;
        height: 100%;
    }

    #right-sidebar {
        width: 30;
        height: 100%;
        border-left: thick $background-lighten-2;
        background: $surface;
        padding: 1;
        display: none;
    }

    #right-sidebar.-visible {
        display: block;
    }

    ClickableBox {
        layout: stream;
        width: 10;
        height: 5;
        background: $secondary;
        border: tall $secondary-darken-2;
        transition: border 150ms in_out_cubic;
    }

    ClickableBox:hover {
        border: tall $secondary-lighten-2;
    }
    
    /* CSS for the new directory modal dialog */
    NewDirectoryScreen {
        align: center middle;
    }

    #dialog {
        padding: 0 1;
        background: $surface;
        width: 60;
        height: auto;
        border: thick $primary-lighten-2;
    }

    #dialog-title {
        width: 100%;
        align: center middle;
        text-style: bold;
        padding-top: 1;
    }
    
    .new-dir-label {
        margin: 1 0 1 1;
    }
    
    #new-dir-input {
        margin: 0 1;
    }

    .buttons {
        width: 100%;
        align-horizontal: right;
        padding-top: 1;
    }

    .buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
    ]

    show_right_sidebar = reactive(False)

    def watch_show_right_sidebar(self, show: bool) -> None:
        self.query_one("#right-sidebar").set_class(show, "-visible")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal(id="main-container"):
            # The left sidebar now contains the tree and the new button
            with Container(id="left-sidebar"):
                yield DirectoryTree(str(Path.cwd()), id="tree")
                yield Button("Add Directory", id="add-dir-button", variant="success")
            yield Container(id="main-area")
            yield Container(
                Static("Box Properties", id="properties-text"), id="right-sidebar"
            )
        yield Footer()

    def on_click(self, event: events.Click) -> None:
        main_area = self.query_one("#main-area")
        if main_area.content_region.contains(event.screen_x, event.screen_y):
            box = ClickableBox()
            box.styles.offset = (
                event.x - main_area.content_region.x,
                event.y - main_area.content_region.y,
            )
            main_area.mount(box)

    def on_clickable_box_clicked(self, message: ClickableBox.Clicked) -> None:
        self.show_right_sidebar = not self.show_right_sidebar
        properties_panel = self.query_one("#properties-text", Static)
        if self.show_right_sidebar:
            clicked_box = message.box
            properties_panel.update(
                f"[b]Box Properties[/b]\n\n"
                f"Offset: ({int(clicked_box.styles.offset.x.value)}, {int(clicked_box.styles.offset.y.value)})"
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when the 'Add Directory' button is pressed."""
        if event.button.id == "add-dir-button":

            def create_directory(name: str) -> None:
                """Callback function to create directory after modal closes."""
                if not name:
                    # User cancelled or entered an empty name
                    return

                tree = self.query_one(DirectoryTree)
                # If no item is selected in the tree, we can't create a directory
                if tree.cursor_node is None:
                    self.log.warning("No directory selected in the tree.")
                    self.bell()  # Audible feedback for the user
                    return

                # Determine the parent path for the new directory
                selected_path = Path(tree.cursor_node.data.path)
                parent_path = (
                    selected_path if selected_path.is_dir() else selected_path.parent
                )

                new_dir_path = parent_path / name
                try:
                    # Create the directory on the actual filesystem
                    os.mkdir(new_dir_path)
                except (OSError, FileExistsError) as e:
                    # In a real app, you might show a proper error dialog here
                    self.log.error(f"Error creating directory: {e}")
                    self.bell()
                else:
                    self.log.info(f"Directory created: {new_dir_path}")
                    # Reload the tree to show the new directory
                    tree.reload()

            # Push the modal screen and run the callback when it's dismissed
            await self.push_screen(NewDirectoryScreen(), create_directory)


if __name__ == "__main__":
    app = DirectoryBoxApp()
    app.run()
