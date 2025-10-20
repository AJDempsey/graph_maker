import math
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input
from rich.text import Text


def _get_connection_points(
    cx: int, cy: int, width: int, height: int
) -> list[tuple[int, int]]:
    """Returns the 8 connection points (+) for a node."""
    h_offset_mid = 0
    h_offset_outer = width // 2  # 3
    v_offset_mid = 0
    v_offset_outer = height // 2  # 3
    return [
        (cx - h_offset_outer, cy - v_offset_outer),  # top-left
        (cx + h_offset_mid, cy - v_offset_outer),  # top-middle
        (cx + h_offset_outer, cy - v_offset_outer),  # top-right
        (cx - h_offset_outer, cy + v_offset_mid),  # middle-left
        (cx + h_offset_outer, cy + v_offset_mid),  # middle-right
        (cx - h_offset_outer, cy + v_offset_outer),  # bottom-left
        (cx + h_offset_mid, cy + v_offset_outer),  # bottom-middle
        (cx + h_offset_outer, cy + v_offset_outer),  # bottom-right
    ]


def get_line_points(x0: int, y0: int, x1: int, y1: int):
    """
    Generator for coordinates on a line using Bresenham's algorithm.
    Yields (x, y) tuples.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def get_snapped_line_endpoints(
    p1: tuple[int, int],
    p2: tuple[int, int],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Calculates the start/end points of a line by finding the closest connection points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if (x1, y1) == (x2, y2):
        return p1, p2

    h_offset_outer = width // 2
    v_offset_outer = height // 2

    # Case 1: Perfectly horizontal line
    if dy == 0:
        start_point = (x1 + h_offset_outer * (1 if dx > 0 else -1), y1)
        end_point = (x2 - h_offset_outer * (1 if dx > 0 else -1), y2)
        return start_point, end_point

    # Case 2: Perfectly vertical line
    if dx == 0:
        start_point = (x1, y1 + v_offset_outer * (1 if dy > 0 else -1))
        end_point = (x2, y2 - v_offset_outer * (1 if dy > 0 else -1))
        return start_point, end_point

    # Case 3: Diagonal line. Snap one step diagonally outside the corner.
    h_offset_outside = width // 2 + 1
    v_offset_outside = height // 2 + 1

    # Determine the sign of the direction vector for snapping
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1

    start_point = (x1 + sx * h_offset_outside, y1 + sy * v_offset_outside)
    # For the end point, we move from its center in the opposite direction
    end_point = (x2 - sx * h_offset_outside, y2 - sy * v_offset_outside)

    return start_point, end_point


class GraphCanvas(Widget):
    """A widget for drawing and interacting with a graph."""

    class NodesChanged(Message):
        """Posted when the nodes list is modified."""

        pass

    class RenameNode(Message):
        """Posted when a user wants to rename a node."""

        def __init__(self, node_index: int, screen_pos: tuple[int, int]):
            self.node_index = node_index
            self.screen_pos = screen_pos
            super().__init__()

    nodes: reactive[list[dict]] = reactive([])
    edges: reactive[list[tuple[int, int]]] = reactive([])

    zoom: reactive[float] = reactive(1.0)
    pan_x: reactive[float] = reactive(0.0)
    pan_y: reactive[float] = reactive(0.0)

    NODE_WIDTH = 7
    NODE_HEIGHT = 7

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_dragging = False
        self._drag_start_node_idx: int | None = None
        self._drag_current_pos: tuple[float, float] | None = None
        self._initial_center_done = False

    def on_resize(self, event) -> None:
        """Center the initial graph after the first layout."""
        if self.nodes and not self._initial_center_done:
            self.post_message(self.NodesChanged())
            self._initial_center_done = True

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        """Converts screen coordinates to world coordinates."""
        return (sx / self.zoom + self.pan_x, sy / self.zoom + self.pan_y)

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Converts world coordinates to screen coordinates."""
        return (int((wx - self.pan_x) * self.zoom), int((wy - self.pan_y) * self.zoom))

    def _zoom(self, factor: float) -> None:
        """Zooms the canvas, keeping the center point fixed."""
        if not self.size.width or not self.size.height:
            return  # Avoid division by zero if widget not laid out

        center_sx = self.size.width / 2
        center_sy = self.size.height / 2

        # World coordinates at the center of the screen before zoom
        world_center_x, world_center_y = self.screen_to_world(center_sx, center_sy)

        # Apply zoom
        new_zoom = self.zoom * factor

        # Calculate new pan to keep the world center at the screen center
        # Formula: new_pan_x = world_x - (screen_x / new_zoom)
        self.pan_x = world_center_x - (center_sx / new_zoom)
        self.pan_y = world_center_y - (center_sy / new_zoom)
        self.zoom = new_zoom

    def zoom_in(self) -> None:
        """Increases the zoom level, keeping the center fixed."""
        self._zoom(1.25)

    def zoom_out(self) -> None:
        """Decreases the zoom level, keeping the center fixed."""
        self._zoom(1 / 1.25)

    def on_mouse_scroll_up(self, event) -> None:
        """Handle mouse wheel scroll up for zooming in."""
        self.zoom_in()

    def on_mouse_scroll_down(self, event) -> None:
        """Handle mouse wheel scroll down for zooming out."""
        self.zoom_out()

    def _find_node_at(self, wx: float, wy: float) -> int | None:
        """Find the index of a node at a given world coordinate."""
        h_offset = self.NODE_WIDTH / 2
        v_offset = self.NODE_HEIGHT / 2
        for i, node_data in enumerate(self.nodes):
            nx, ny = node_data["pos"]
            if (nx - h_offset <= wx <= nx + h_offset) and (
                ny - v_offset <= wy <= ny + v_offset
            ):
                return i
        return None

    def on_mouse_down(self, event) -> None:
        """Handle mouse down events."""
        world_x, world_y = self.screen_to_world(event.x, event.y)
        node_idx = self._find_node_at(world_x, world_y)

        # Ctrl + Left-click to rename a node
        if event.ctrl and event.button == 1:
            if node_idx is not None:
                screen_pos = self.world_to_screen(*self.nodes[node_idx]["pos"])
                self.post_message(self.RenameNode(node_idx, screen_pos))
            return

        # Right-click to delete a node
        if event.button == 3:
            if node_idx is not None:
                self._delete_node(node_idx)
            return

        # Left-click to add a node or start dragging
        if node_idx is not None:
            self._is_dragging = True
            self._drag_start_node_idx = node_idx
            self._drag_current_pos = (world_x, world_y)
            self.refresh()
        else:
            new_node = {"pos": (world_x, world_y), "name": str(len(self.nodes))}
            can_add = True
            for node_data in self.nodes:
                nx, ny = node_data["pos"]
                if (
                    abs(new_node["pos"][0] - nx) < self.NODE_WIDTH
                    and abs(new_node["pos"][1] - ny) < self.NODE_HEIGHT
                ):
                    can_add = False
                    break
            if can_add:
                self.nodes = self.nodes + [new_node]
                self.post_message(self.NodesChanged())

    def on_mouse_move(self, event) -> None:
        """Handle mouse movement for dragging or panning."""
        if self._is_dragging:
            self._drag_current_pos = self.screen_to_world(event.x, event.y)
            self.refresh()

    def on_mouse_up(self, event) -> None:
        """Handle mouse up events to complete an action."""
        if self._is_dragging:
            world_x, world_y = self.screen_to_world(event.x, event.y)
            target_node_idx = self._find_node_at(world_x, world_y)
            if (
                target_node_idx is not None
                and self._drag_start_node_idx is not None
                and target_node_idx != self._drag_start_node_idx
            ):
                start_idx = min(self._drag_start_node_idx, target_node_idx)
                end_idx = max(self._drag_start_node_idx, target_node_idx)
                new_edge = (start_idx, end_idx)
                if new_edge not in self.edges:
                    self.edges = self.edges + [new_edge]

            self._is_dragging = False
            self._drag_start_node_idx = None
            self._drag_current_pos = None
            self.refresh()

    def _delete_node(self, delete_idx: int) -> None:
        """Deletes a node and its connected edges, then relabels."""
        original_nodes = self.nodes
        new_nodes = [
            node.copy() for i, node in enumerate(original_nodes) if i != delete_idx
        ]

        # Relabel nodes, preserving custom names.
        # A name is "custom" if it wasn't the string representation of its original index.
        for i, node in enumerate(new_nodes):
            original_idx = i if i < delete_idx else i + 1
            original_node = original_nodes[original_idx]

            if original_node["name"] != str(original_idx):
                # This was a custom name, so we keep it.
                node["name"] = original_node["name"]
            else:
                # This was a default name, so we update it to the new index.
                node["name"] = str(i)

        new_edges = []
        for u, v in self.edges:
            if u == delete_idx or v == delete_idx:
                continue
            new_u = u if u < delete_idx else u - 1
            new_v = v if v < delete_idx else v - 1
            new_edges.append(tuple(sorted((new_u, new_v))))
        self.nodes = new_nodes
        self.edges = list(sorted(list(set(new_edges))))
        self.post_message(self.NodesChanged())

    def _draw_node_box(self, grid, x, y, label: str, width, height):
        """Helper to draw a box representing a node onto a grid."""
        max_y, max_x = len(grid), len(grid[0])

        # Define the shape relative to the center (x, y)
        shape = [
            # (dx, dy, char)
            # Top/Bottom rows
            (-3, -3, "+"),
            (-2, -3, "-"),
            (-1, -3, "-"),
            (0, -3, "+"),
            (1, -3, "-"),
            (2, -3, "-"),
            (3, -3, "+"),
            (-3, 3, "+"),
            (-2, 3, "-"),
            (-1, 3, "-"),
            (0, 3, "+"),
            (1, 3, "-"),
            (2, 3, "-"),
            (3, 3, "+"),
            # Middle "+" connection points
            (-3, 0, "+"),
            (3, 0, "+"),
            # Vertical bars
            (-3, -2, "|"),
            (-3, -1, "|"),
            (-3, 1, "|"),
            (-3, 2, "|"),
            (3, -2, "|"),
            (3, -1, "|"),
            (3, 1, "|"),
            (3, 2, "|"),
        ]

        for dx, dy, char in shape:
            col, row = x + dx, y + dy
            if 0 <= row < max_y and 0 <= col < max_x:
                grid[row][col] = char

        # Draw the label centered inside the box
        if len(label) > 0 and width > 2 and height > 2:
            label_len = len(label)
            max_label_len = width - 2
            if label_len > max_label_len:
                label = label[:max_label_len]
                label_len = max_label_len

            label_start_col = x - label_len // 2
            for i, char in enumerate(label):
                col = label_start_col + i
                if 0 <= y < max_y and 0 <= col < max_x:
                    if grid[y][col] == " ":
                        grid[y][col] = char

    def render(self) -> Text:
        """Render the graph onto the widget's canvas."""
        if not self.size.width or not self.size.height:
            return Text("")

        grid = [[" " for _ in range(self.size.width)] for _ in range(self.size.height)]

        # Draw nodes first, so edges can draw over the corners.
        node_w_screen = max(3, int(self.NODE_WIDTH * self.zoom))
        node_h_screen = max(3, int(self.NODE_HEIGHT * self.zoom))
        for i, node_data in enumerate(self.nodes):
            wx, wy = node_data["pos"]
            sx, sy = self.world_to_screen(wx, wy)
            self._draw_node_box(
                grid, sx, sy, node_data["name"], node_w_screen, node_h_screen
            )

        # Draw committed edges (snapped)
        for start_idx, end_idx in self.edges:
            p1_world = self.nodes[start_idx]["pos"]
            p2_world = self.nodes[end_idx]["pos"]
            sp1_world, sp2_world = get_snapped_line_endpoints(
                p1_world, p2_world, self.NODE_WIDTH, self.NODE_HEIGHT
            )
            sp1_screen = self.world_to_screen(*sp1_world)
            sp2_screen = self.world_to_screen(*sp2_world)

            for x, y in get_line_points(
                sp1_screen[0], sp1_screen[1], sp2_screen[0], sp2_screen[1]
            ):
                if 0 <= y < self.size.height and 0 <= x < self.size.width:
                    # Only draw on empty space.
                    if grid[y][x] == " ":
                        grid[y][x] = "."

        # Draw the line being currently dragged (free-draw)
        if (
            self._is_dragging
            and self._drag_start_node_idx is not None
            and self._drag_current_pos
        ):
            start_node_pos_w = self.nodes[self._drag_start_node_idx]["pos"]
            start_sx, start_sy = self.world_to_screen(*start_node_pos_w)
            end_sx, end_sy = self.world_to_screen(*self._drag_current_pos)
            for x, y in get_line_points(start_sx, start_sy, end_sx, end_sy):
                if 0 <= y < self.size.height and 0 <= x < self.size.width:
                    # Only draw on empty space.
                    if grid[y][x] == " ":
                        grid[y][x] = ":"

        return Text("\n".join("".join(row) for row in grid))


class AsciiGraphApp(App):
    """A Textual app to create and export ASCII graphs."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-container {
        height: 1fr;
        layout: vertical;
        border: heavy $primary;
        padding: 1;
    }
    GraphCanvas {
        width: 1fr;
        height: 1fr;
        border: solid $accent;
    }
    #button-container {
        height: auto;
        align: center middle;
        padding-top: 1;
    }
    #node-name-input {
        display: none;
        position: absolute;
        width: 12;
        height: 3;
        layer: top;
    }
    """

    BINDINGS = [
        ("c", "clear_canvas", "Clear"),
        ("q", "quit", "Quit"),
        ("plus", "zoom_in", "Zoom In"),
        ("minus", "zoom_out", "Zoom Out"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._node_to_rename_idx: int | None = None

    def on_mount(self) -> None:
        """Set up the initial state of the app with one node."""
        # The app starts with a single node. The exact coordinates don't matter
        # as the centering logic in on_graph_canvas_nodes_changed will place it correctly.
        self.query_one(GraphCanvas).nodes = [{"pos": (0.0, 0.0), "name": "0"}]

    def action_zoom_in(self) -> None:
        """Zoom in on the canvas."""
        self.query_one(GraphCanvas).zoom_in()

    def action_zoom_out(self) -> None:
        """Zoom out on the canvas."""
        self.query_one(GraphCanvas).zoom_out()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            GraphCanvas(),
            Container(
                Button("Commit to README.md", variant="success", id="commit"),
                id="button-container",
            ),
            Input(id="node-name-input", placeholder="Name"),
            id="main-container",
        )
        yield Footer()

    def action_clear_canvas(self) -> None:
        """Clear all nodes and edges from the canvas."""
        canvas = self.query_one(GraphCanvas)
        canvas.nodes = []
        canvas.edges = []
        canvas.pan_x = 0
        canvas.pan_y = 0
        canvas.zoom = 1.0
        self.notify("Canvas cleared.")

    def _symmetrize_layout(
        self,
        nodes_to_process: list[dict],
        node_width: int,
        node_height: int,
    ) -> list[dict]:
        """Aligns and symmetrizes a list of nodes."""
        nodes = [n.copy() for n in nodes_to_process]
        node_count = len(nodes)
        h_width = node_width / 2
        h_height = node_height / 2

        # 1. Align X coordinates
        adj_x = [[] for _ in range(node_count)]
        for i in range(node_count):
            for j in range(i + 1, node_count):
                ix, _ = nodes[i]["pos"]
                jx, _ = nodes[j]["pos"]
                if (ix - h_width) < (jx + h_width) and (jx - h_width) < (ix + h_width):
                    adj_x[i].append(j)
                    adj_x[j].append(i)
        visited = [False] * node_count
        for i in range(node_count):
            if not visited[i]:
                group, q, head = [], [i], 0
                visited[i] = True
                while head < len(q):
                    u = q[head]
                    head += 1
                    group.append(u)
                    for v in adj_x[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                avg_x = sum(nodes[n_idx]["pos"][0] for n_idx in group) / len(group)
                for n_idx in group:
                    nodes[n_idx]["pos"] = (int(avg_x), nodes[n_idx]["pos"][1])

        # 2. Align Y coordinates
        adj_y = [[] for _ in range(node_count)]
        for i in range(node_count):
            for j in range(i + 1, node_count):
                _, iy = nodes[i]["pos"]
                _, jy = nodes[j]["pos"]
                if (iy - h_height) < (jy + h_height) and (jy - h_height) < (
                    iy + h_height
                ):
                    adj_y[i].append(j)
                    adj_y[j].append(i)
        visited = [False] * node_count
        for i in range(node_count):
            if not visited[i]:
                group, q, head = [], [i], 0
                visited[i] = True
                while head < len(q):
                    u = q[head]
                    head += 1
                    group.append(u)
                    for v in adj_y[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                avg_y = sum(nodes[n_idx]["pos"][1] for n_idx in group) / len(group)
                for n_idx in group:
                    nodes[n_idx]["pos"] = (nodes[n_idx]["pos"][0], int(avg_y))

        # 3. & 4. Symmetrize and unify spacing for a fixed grid layout
        unique_x_coords = sorted(list({n["pos"][0] for n in nodes}))
        unique_y_coords = sorted(list({n["pos"][1] for n in nodes}))

        # A gap of 7 cells between nodes means the distance between centers
        # is node_width + 7. This results in a diagonal line length of 7.
        unified_gap = node_width + 7

        # Rebuild the X coordinates using the fixed unified gap
        if len(unique_x_coords) > 1:
            n_cols = len(unique_x_coords)
            total_width = (n_cols - 1) * unified_gap
            center_x = (unique_x_coords[0] + unique_x_coords[-1]) / 2.0
            new_start_x = center_x - total_width / 2.0

            x_map = {
                old_x: int(new_start_x + i * unified_gap)
                for i, old_x in enumerate(unique_x_coords)
            }
            nodes = [
                {"pos": (x_map[n["pos"][0]], n["pos"][1]), "name": n["name"]}
                for n in nodes
            ]

        # Rebuild the Y coordinates using the fixed unified gap
        if len(unique_y_coords) > 1:
            n_rows = len(unique_y_coords)
            total_height = (n_rows - 1) * unified_gap
            center_y = (unique_y_coords[0] + unique_y_coords[-1]) / 2.0
            new_start_y = center_y - total_height / 2.0

            y_map = {
                old_y: int(new_start_y + i * unified_gap)
                for i, old_y in enumerate(unique_y_coords)
            }
            nodes = [
                {"pos": (n["pos"][0], y_map[n["pos"][1]]), "name": n["name"]}
                for n in nodes
            ]
        return nodes

    def on_graph_canvas_nodes_changed(self, event: GraphCanvas.NodesChanged) -> None:
        """Handle live symmetrization and centering when the graph's nodes change."""
        canvas = self.query_one(GraphCanvas)
        if not canvas.nodes:
            # If no nodes, reset pan
            canvas.pan_x = 0
            canvas.pan_y = 0
            return

        symmetrized_nodes = self._symmetrize_layout(
            canvas.nodes, canvas.NODE_WIDTH, canvas.NODE_HEIGHT
        )
        canvas.nodes = symmetrized_nodes

        # --- Recenter the graph in the view ---
        if symmetrized_nodes and canvas.size.width > 0 and canvas.size.height > 0:
            min_x = min(n["pos"][0] for n in symmetrized_nodes)
            max_x = max(n["pos"][0] for n in symmetrized_nodes)
            min_y = min(n["pos"][1] for n in symmetrized_nodes)
            max_y = max(n["pos"][1] for n in symmetrized_nodes)

            graph_center_x = (min_x + max_x) / 2
            graph_center_y = (min_y + max_y) / 2

            screen_center_x = canvas.size.width / 2
            screen_center_y = canvas.size.height / 2

            # Adjust pan to move the graph's center to the screen's center
            canvas.pan_x = graph_center_x - (screen_center_x / canvas.zoom)
            canvas.pan_y = graph_center_y - (screen_center_y / canvas.zoom)

    async def on_graph_canvas_rename_node(
        self, message: GraphCanvas.RenameNode
    ) -> None:
        """Handle the request to rename a node."""
        input_widget = self.query_one("#node-name-input", Input)
        self._node_to_rename_idx = message.node_index
        canvas = self.query_one(GraphCanvas)

        input_widget.styles.display = "block"
        input_widget.styles.offset = message.screen_pos
        current_name = canvas.nodes[message.node_index]["name"]
        input_widget.value = current_name
        input_widget.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle the submission of the node name input."""
        if event.input.id == "node-name-input":
            new_name = event.value
            event.input.styles.display = "none"

            canvas = self.query_one(GraphCanvas)
            if self._node_to_rename_idx is not None:
                # Create a new list of new dictionaries to ensure reactive update
                new_nodes_list = [n.copy() for n in canvas.nodes]
                new_nodes_list[self._node_to_rename_idx]["name"] = new_name
                canvas.nodes = new_nodes_list
                self._node_to_rename_idx = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the commit button press."""
        if event.button.id == "commit":
            canvas = self.query_one(GraphCanvas)
            # Nodes are already symmetrized from the live updates
            nodes = canvas.nodes
            edges = canvas.edges

            if not nodes:
                self.notify(
                    "Canvas is empty. Add some nodes first.", severity="warning"
                )
                return

            # --- Generate ASCII representation ---
            padding = canvas.NODE_WIDTH
            if not nodes:
                min_x, max_x, min_y, max_y = 0, 0, 0, 0
            else:
                min_x = min(n["pos"][0] for n in nodes)
                max_x = max(n["pos"][0] for n in nodes)
                min_y = min(n["pos"][1] for n in nodes)
                max_y = max(n["pos"][1] for n in nodes)

            width = int(max_x - min_x + padding * 2)
            height = int(max_y - min_y + padding * 2)
            offset_x = -min_x + padding
            offset_y = -min_y + padding

            width = max(1, width)
            height = max(1, height)

            grid = [[" " for _ in range(width)] for _ in range(height)]

            # Draw nodes first, so edges can draw over them.
            for i, node_data in enumerate(nodes):
                nx, ny = node_data["pos"]
                canvas._draw_node_box(
                    grid,
                    int(nx + offset_x),
                    int(ny + offset_y),
                    node_data["name"],
                    canvas.NODE_WIDTH,
                    canvas.NODE_HEIGHT,
                )

            # Draw edges
            for start_idx, end_idx in edges:
                p1 = nodes[start_idx]["pos"]
                p2 = nodes[end_idx]["pos"]

                p1_grid = (p1[0] + offset_x, p1[1] + offset_y)
                p2_grid = (p2[0] + offset_x, p2[1] + offset_y)

                sp1, sp2 = get_snapped_line_endpoints(
                    p1_grid, p2_grid, canvas.NODE_WIDTH, canvas.NODE_HEIGHT
                )

                dx = sp2[0] - sp1[0]
                dy = sp2[1] - sp1[1]

                is_diagonal = dx != 0 and dy != 0
                if is_diagonal:
                    char = "\\" if (dx > 0) == (dy > 0) else "/"
                elif dx != 0:  # Horizontal
                    char = "-"
                else:  # Vertical
                    char = "|"

                for x0, y0 in get_line_points(sp1[0], sp1[1], sp2[0], sp2[1]):
                    if 0 <= y0 < height and 0 <= x0 < width:
                        # Only draw on empty space to avoid overwriting node corners.
                        if grid[y0][x0] == " ":
                            grid[y0][x0] = char

            ascii_art = "\n".join("".join(row) for row in grid)

            # --- Write to file ---
            try:
                with open("README.md", "w") as f:
                    f.write("# ASCII Graph\n\n")
                    f.write("Generated by Textual ASCII Graph Editor.\n\n")
                    f.write("```\n")
                    f.write(ascii_art)
                    f.write("\n```\n")
                self.notify("Graph successfully written to README.md", title="Success")
            except Exception as e:
                self.notify(
                    f"Error writing to file: {e}", severity="error", title="Error"
                )


if __name__ == "__main__":
    app = AsciiGraphApp()
    app.run()
