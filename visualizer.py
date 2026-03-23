"""
visualiser.py - DARTS Module 5 live map visualizer

This module runs as a standalone process and renders live simulation state by
reading shared JSON artifacts:
- udupi_graph.graphml
- data/world_state.json
- data/vehicle_states.json
- data/current_graph.json
- data/explanation_log.txt (optional but expected)

It does not import or modify other modules.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

try:
    import osmnx as ox
except ImportError as exc:
    raise ImportError(
        "Module 5 requires osmnx. Install with: pip install osmnx"
    ) from exc


CLOSED_EDGE_WEIGHT = 999999.0
DEFAULT_INTERVAL_MS = 700
VEHICLE_COLORS = {
    "ambulance": "#d62828",
    "truck": "#f77f00",
}
STATUS_MARKER_EDGE = {
    "idle": "#1b4332",
    "en_route": "#2a9d8f",
    "arrived": "#264653",
    "returning": "#6a4c93",
}
INVENTORY_COLORS = {
    "mre": "#2a9d8f",
    "medicine": "#e63946",
    "comms": "#457b9d",
    "ambulances": "#f4a261",
}


class VisualiserError(RuntimeError):
    """Raised when required Module 5 input artifacts are missing or invalid."""


class DartsVisualiser:
    def __init__(self, base_dir: Path, interval_ms: int, save_path: Path | None = None) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.graph_path = base_dir / "udupi_graph.graphml"
        self.world_state_path = self.data_dir / "world_state.json"
        self.vehicle_states_path = self.data_dir / "vehicle_states.json"
        self.current_graph_path = self.data_dir / "current_graph.json"
        self.explanation_log_path = self.data_dir / "explanation_log.txt"
        self.interval_ms = interval_ms
        self.save_path = save_path

        self._preflight()

        self.graph = self._load_graph(self.graph_path)
        self.node_xy = self._extract_node_positions(self.graph)

        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.fig.suptitle("DARTS Live Visualiser", fontsize=15, fontweight="bold")
        self.ax.set_axis_off()

        self._base_edge_collection = self._draw_base_graph()
        self._setup_legend()

        self._dynamic_artists: list[Any] = []
        self._last_rendered_timestep: int | None = None
        self._status_text = self.ax.text(
            0.01,
            0.01,
            "Waiting for first timestep...",
            transform=self.ax.transAxes,
            fontsize=9,
            color="#202020",
            bbox={"facecolor": "#ffffff", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

    def _preflight(self) -> None:
        missing = []
        for p in [
            self.graph_path,
            self.world_state_path,
            self.vehicle_states_path,
            self.current_graph_path,
        ]:
            if not p.exists():
                missing.append(str(p))

        if missing:
            message = "\n".join(f"- {m}" for m in missing)
            raise VisualiserError(
                "Module 5 startup failed. Missing required inputs:\n"
                f"{message}\n"
                "Expected producers: graph_builder.py, simulation_engine.py, vehicle_agent.py"
            )

        self._load_json(self.world_state_path, "world_state")
        self._load_json(self.vehicle_states_path, "vehicle_states")
        self._load_json(self.current_graph_path, "current_graph")

    @staticmethod
    def _load_json(path: Path, label: str) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise VisualiserError(f"Invalid JSON in {label}: {path} ({exc})") from exc
        except OSError as exc:
            raise VisualiserError(f"Failed reading {label}: {path} ({exc})") from exc

        if not isinstance(data, dict):
            raise VisualiserError(f"Expected object JSON for {label}: {path}")
        return data

    @staticmethod
    def _load_graph(path: Path) -> nx.MultiDiGraph:
        try:
            graph = ox.load_graphml(path)
        except Exception as exc:
            raise VisualiserError(f"Could not load graphml file: {path} ({exc})") from exc

        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            raise VisualiserError(f"Graph is empty: {path}")
        return graph

    @staticmethod
    def _extract_node_positions(graph: nx.MultiDiGraph) -> dict[Any, tuple[float, float]]:
        node_xy: dict[Any, tuple[float, float]] = {}
        for node_id, attrs in graph.nodes(data=True):
            if "x" not in attrs or "y" not in attrs:
                continue
            node_xy[node_id] = (float(attrs["x"]), float(attrs["y"]))

        if not node_xy:
            raise VisualiserError("Graph nodes do not contain x/y coordinates")
        return node_xy

    def _draw_base_graph(self) -> LineCollection:
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for u, v, _key in self.graph.edges(keys=True):
            if u not in self.node_xy or v not in self.node_xy:
                continue
            segments.append((self.node_xy[u], self.node_xy[v]))

        if not segments:
            raise VisualiserError("Unable to draw base graph edges from graphml")

        line_collection = LineCollection(segments, linewidths=0.5, colors="#9aa0a6", alpha=0.55)
        self.ax.add_collection(line_collection)

        x_values = [x for x, _y in self.node_xy.values()]
        y_values = [y for _x, y in self.node_xy.values()]
        self.ax.set_xlim(min(x_values), max(x_values))
        self.ax.set_ylim(min(y_values), max(y_values))
        self.ax.set_aspect("equal", adjustable="box")
        return line_collection

    def _setup_legend(self) -> None:
        legend_handles = [
            plt.Line2D([], [], marker="o", color="w", markerfacecolor=VEHICLE_COLORS["ambulance"], label="Ambulance", markersize=8),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor=VEHICLE_COLORS["truck"], label="Truck", markersize=8),
            plt.Line2D([], [], color="#e63946", linewidth=2.2, label="Closed edge"),
            plt.Line2D([], [], color="#9aa0a6", linewidth=1.5, label="Passable road"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    def _resolve_graph_node(self, raw_node: Any) -> Any:
        if raw_node in self.node_xy:
            return raw_node

        as_str = str(raw_node)
        if as_str in self.node_xy:
            return as_str

        try:
            as_int = int(as_str)
        except ValueError:
            as_int = None
        if as_int is not None and as_int in self.node_xy:
            return as_int

        raise VisualiserError(
            "Vehicle/node reference is not a graph node id: "
            f"{raw_node!r}. Ensure Module 2/3 emit graph-compatible node ids."
        )

    @staticmethod
    def _parse_edge_tuple(edge_str: str) -> tuple[Any, Any] | None:
        try:
            parsed = ast.literal_eval(edge_str)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
            return None
        return parsed[0], parsed[1]

    def _extract_closed_edges(self, current_graph: dict[str, Any]) -> list[tuple[Any, Any]]:
        closed: list[tuple[Any, Any]] = []

        weights = current_graph.get("weights")
        if isinstance(weights, dict):
            for edge_str, weight in weights.items():
                try:
                    numeric_weight = float(weight)
                except (TypeError, ValueError):
                    continue
                if numeric_weight < CLOSED_EDGE_WEIGHT:
                    continue
                edge_tuple = self._parse_edge_tuple(str(edge_str))
                if not edge_tuple:
                    continue
                closed.append(edge_tuple)

        edges = current_graph.get("edges")
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                try:
                    numeric_weight = float(edge.get("weight", 1.0))
                except (TypeError, ValueError):
                    continue
                if numeric_weight < CLOSED_EDGE_WEIGHT:
                    continue
                if "u" in edge and "v" in edge:
                    closed.append((edge["u"], edge["v"]))

        unique = []
        seen = set()
        for u, v in closed:
            key = (str(u), str(v))
            if key in seen:
                continue
            seen.add(key)
            unique.append((u, v))
        return unique

    def _read_explanation_tail(self) -> str:
        if not self.explanation_log_path.exists():
            return "No explanation log found"

        try:
            lines = self.explanation_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "Explanation log unreadable"

        for line in reversed(lines):
            candidate = line.strip()
            if not candidate:
                continue
            if candidate.startswith("="):
                continue
            return candidate
        return "No explanation available"

    def _clear_dynamic_artists(self) -> None:
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._dynamic_artists = []

    def _draw_closed_edges(self, closed_edges: list[tuple[Any, Any]]) -> None:
        segments = []
        for raw_u, raw_v in closed_edges:
            try:
                u = self._resolve_graph_node(raw_u)
                v = self._resolve_graph_node(raw_v)
            except VisualiserError:
                continue
            if u not in self.node_xy or v not in self.node_xy:
                continue
            segments.append((self.node_xy[u], self.node_xy[v]))

        if not segments:
            return

        collection = LineCollection(segments, linewidths=2.4, colors="#e63946", alpha=0.9)
        self.ax.add_collection(collection)
        self._dynamic_artists.append(collection)

    def _draw_vehicles(self, vehicles: list[dict[str, Any]]) -> None:
        for vehicle in vehicles:
            if "id" not in vehicle or "position" not in vehicle:
                raise VisualiserError("vehicle_states entry missing required fields: id/position")

            node_id = self._resolve_graph_node(vehicle["position"])
            if node_id not in self.node_xy:
                continue

            x, y = self.node_xy[node_id]
            vtype = str(vehicle.get("type", "truck")).lower()
            color = VEHICLE_COLORS.get(vtype, "#f77f00")
            status = str(vehicle.get("status", "idle")).lower()
            edgecolor = STATUS_MARKER_EDGE.get(status, "#333333")

            point = self.ax.scatter(
                [x],
                [y],
                c=color,
                s=65,
                edgecolors=edgecolor,
                linewidths=1.1,
                zorder=6,
            )
            self._dynamic_artists.append(point)

            label = self.ax.text(
                x,
                y,
                f" {vehicle['id']}",
                fontsize=8,
                color="#202020",
                zorder=7,
            )
            self._dynamic_artists.append(label)

            route_remaining = vehicle.get("route_remaining", [])
            if isinstance(route_remaining, list) and route_remaining:
                route_nodes = [node_id]
                for raw in route_remaining:
                    route_nodes.append(self._resolve_graph_node(raw))
                coords = [self.node_xy[rn] for rn in route_nodes if rn in self.node_xy]
                if len(coords) >= 2:
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    route_line = self.ax.plot(
                        xs,
                        ys,
                        linestyle="--",
                        linewidth=1.0,
                        color="#f4a261",
                        alpha=0.8,
                        zorder=5,
                    )[0]
                    self._dynamic_artists.append(route_line)

    def _draw_inventory_bars(self, world_state: dict[str, Any]) -> None:
        depot_inventory = world_state.get("depot_inventory")
        if not isinstance(depot_inventory, dict):
            raise VisualiserError("world_state missing depot_inventory object")

        vehicles_by_depot: dict[str, Any] = {}
        vehicle_state = self._load_json(self.vehicle_states_path, "vehicle_states")
        vehicles_list = self._extract_vehicle_list(vehicle_state)
        for vehicle in vehicles_list:
            depot = vehicle.get("depot")
            position = vehicle.get("position")
            if depot and position and depot not in vehicles_by_depot:
                vehicles_by_depot[str(depot)] = position

        for depot_id, commodities in depot_inventory.items():
            if str(depot_id) not in vehicles_by_depot:
                continue

            graph_node = self._resolve_graph_node(vehicles_by_depot[str(depot_id)])
            x, y = self.node_xy[graph_node]

            if not isinstance(commodities, dict):
                continue

            values = {
                "mre": int(commodities.get("mre", 0)),
                "medicine": int(commodities.get("medicine", 0)),
                "comms": int(commodities.get("comms", 0)),
                "ambulances": int(commodities.get("ambulances", 0)),
            }
            max_value = max(max(values.values()), 1)

            bar_width = 0.0012
            bar_gap = 0.00045
            base_x = x + 0.001
            base_y = y + 0.0004
            bar_height_max = 0.0025

            for idx, commodity in enumerate(["mre", "medicine", "comms", "ambulances"]):
                height = bar_height_max * (values[commodity] / max_value)
                rect = Rectangle(
                    (base_x + idx * (bar_width + bar_gap), base_y),
                    bar_width,
                    height,
                    facecolor=INVENTORY_COLORS[commodity],
                    edgecolor="#3d3d3d",
                    linewidth=0.35,
                    alpha=0.9,
                    zorder=6,
                )
                self.ax.add_patch(rect)
                self._dynamic_artists.append(rect)

            depot_label = self.ax.text(
                base_x,
                base_y + bar_height_max + 0.00035,
                str(depot_id),
                fontsize=7,
                color="#1d3557",
                zorder=7,
            )
            self._dynamic_artists.append(depot_label)

    @staticmethod
    def _extract_vehicle_list(vehicle_states: dict[str, Any]) -> list[dict[str, Any]]:
        if "vehicles" in vehicle_states and isinstance(vehicle_states["vehicles"], list):
            return [v for v in vehicle_states["vehicles"] if isinstance(v, dict)]
        if isinstance(vehicle_states.get("vehicle_status"), list):
            return [v for v in vehicle_states["vehicle_status"] if isinstance(v, dict)]
        raise VisualiserError(
            "vehicle_states JSON must include list under 'vehicles' or 'vehicle_status'"
        )

    def _render_frame(self, _frame_idx: int) -> list[Any]:
        world_state = self._load_json(self.world_state_path, "world_state")
        vehicle_states = self._load_json(self.vehicle_states_path, "vehicle_states")
        current_graph = self._load_json(self.current_graph_path, "current_graph")

        timestep = world_state.get("timestep")
        if not isinstance(timestep, int):
            raise VisualiserError("world_state missing integer 'timestep'")

        if self._last_rendered_timestep is not None and timestep == self._last_rendered_timestep:
            return []

        self._last_rendered_timestep = timestep
        self._clear_dynamic_artists()

        closed_edges = self._extract_closed_edges(current_graph)
        vehicles = self._extract_vehicle_list(vehicle_states)

        self._draw_closed_edges(closed_edges)
        self._draw_vehicles(vehicles)
        self._draw_inventory_bars(world_state)

        explanation = self._read_explanation_tail()
        explanation_box = self.ax.text(
            0.99,
            0.99,
            f"Timestep: {timestep}\nLLM: {explanation}",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            color="#202020",
            bbox={"facecolor": "#ffffff", "edgecolor": "#bbbbbb", "alpha": 0.92},
            zorder=9,
        )
        self._dynamic_artists.append(explanation_box)

        self._status_text.set_text(
            f"Last update: timestep={timestep} | vehicles={len(vehicles)} | closed_edges={len(closed_edges)}"
        )
        return self._dynamic_artists

    def run(self) -> None:
        ani = animation.FuncAnimation(
            self.fig,
            self._render_frame,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            writer = animation.FFMpegWriter(fps=max(1, int(1000 / self.interval_ms)))
            ani.save(self.save_path, writer=writer)
            print(f"Saved animation to: {self.save_path}")
            return

        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DARTS Module 5 live visualiser")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory containing data/ and udupi_graph.graphml",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Polling interval for reloading JSON files",
    )
    parser.add_argument(
        "--save-mp4",
        type=Path,
        default=None,
        help="Optional output path to save mp4 instead of opening live window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        visualiser = DartsVisualiser(
            base_dir=args.base_dir,
            interval_ms=args.interval_ms,
            save_path=args.save_mp4,
        )
        visualiser.run()
        return 0
    except VisualiserError as exc:
        print(f"[MODULE 5 ERROR] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nModule 5 visualiser stopped by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
