import asyncio
import random
from csv import DictWriter
from datetime import datetime
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple

import pygame

from colors import BLACK, BLUE, GREEN, LIGHT_GRAY, RED, WHITE

logger = Logger(__name__)
ROOT_DIR: Path = Path(__file__).parent

### Parameters ###

FONT: pygame.font.Font
FILE: Path

# Screen dimensions
WIDTH: int
HEIGHT: int
GRID_SIZE: int
CELL_HEIGHT: float
CELL_WIDTH: float
TOTAL_VEHICLES: int
BATTERY_DEPLETION_RATE: int
RECHARGE_TIME: int
MAX_USERS: int
USER_PROBABILITY: float
FPS: int
HIGH_DEMAND_ZONE: bool
VEHICLE_INIT: str

### Entities ###


class State(Enum):
    AVAILABLE = "Available"
    IN_USE = "In Use"
    NEEDS_RECHARGING = "Needs Recharging"


class Vehicle:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

        self.battery: int = 100
        self.state: State = State.AVAILABLE
        self.recharge_timer: int = 0

        self.user: User = None

    def update(self):
        if self.state == State.NEEDS_RECHARGING:
            self.recharge_timer += 1
            if self.recharge_timer >= RECHARGE_TIME:
                self.state = State.AVAILABLE
                self.battery = 100
                self.recharge_timer = 0


class User:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

        while True:
            self.goal_x = random.randint(0, GRID_SIZE - 1)
            self.goal_y = random.randint(0, GRID_SIZE - 1)
            if self.goal_x != self.x and self.goal_y != self.y:
                break

        self.vehicle: Vehicle = None
        self.wait_time: int = 0

    def update(self):
        if self.x < self.goal_x:
            self.x += 1
        elif self.x > self.goal_x:
            self.x -= 1

        if self.y < self.goal_y:
            self.y += 1
        elif self.y > self.goal_y:
            self.y -= 1

        if self.vehicle is None:
            self.wait_time += 1
        else:
            self.wait_time = 0

    def has_reached_goal(self) -> bool:
        return self.x == self.goal_x and self.y == self.goal_y


### Simulation ###


def _set_global_parameters(
    width: int,
    height: int,
    grid_size: int,
    total_vehicles: int,
    battery_depletion_rate: int,
    recharge_time: int,
    max_users: int,
    user_probability: float,
    fps: int,
    high_demand_zone: bool,
    vehicle_init: str,
):
    global FONT, FILE, WIDTH, HEIGHT, GRID_SIZE, CELL_HEIGHT, CELL_WIDTH, TOTAL_VEHICLES, BATTERY_DEPLETION_RATE, RECHARGE_TIME, MAX_USERS, USER_PROBABILITY, FPS, HIGH_DEMAND_ZONE, VEHICLE_INIT

    FONT = pygame.font.Font(None, int(width * 0.04))
    FILE = Path(
        ROOT_DIR / f"metrics/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    if not FILE.parent.exists():
        FILE.parent.mkdir(parents=True, exist_ok=True)

    WIDTH = width
    HEIGHT = height
    GRID_SIZE = grid_size
    CELL_HEIGHT = HEIGHT / GRID_SIZE
    CELL_WIDTH = WIDTH / GRID_SIZE
    TOTAL_VEHICLES = total_vehicles
    BATTERY_DEPLETION_RATE = battery_depletion_rate
    RECHARGE_TIME = recharge_time
    MAX_USERS = max_users
    USER_PROBABILITY = user_probability
    FPS = fps
    HIGH_DEMAND_ZONE = high_demand_zone
    VEHICLE_INIT = vehicle_init


def _calculate_metrics(
    vehicles: List[Vehicle], users: List[User], frame: int
) -> Dict[str, float]:
    available_vehicles = sum(1 for v in vehicles if v.state == State.AVAILABLE)
    available_percentage = (available_vehicles / TOTAL_VEHICLES) * 100

    vehicles_depleted = sum(1 for v in vehicles if v.state == State.NEEDS_RECHARGING)

    user_wait_times = [u.wait_time for u in users if u.vehicle is None]
    avg_wait_time = (
        sum(user_wait_times) / len(user_wait_times) if user_wait_times else 0
    )

    utilization_percentage = (
        sum(1 for v in vehicles if v.state == State.IN_USE) / TOTAL_VEHICLES
    ) * 100

    return {
        "frame": frame,
        "users": len(users),
        "vehicles": len(vehicles),
        "average_user_wait_time": avg_wait_time,
        "vehicles_available": available_percentage,
        "vehicles_depleted": vehicles_depleted,
        "vehicle_utilization": utilization_percentage,
    }


def _write_metrics(metrics: Dict[str, float]):
    with open(FILE, "a", newline="") as f:
        writer = DictWriter(f, fieldnames=metrics.keys())
        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(metrics)


def _render_stats_and_back(
    screen: pygame.Surface, metrics: Dict[str, float]
) -> pygame.Rect:
    stats_x = int(WIDTH * 0.02)
    stats_y = int(HEIGHT * 0.02)

    button_width = int(WIDTH * 0.25)
    button_height = int(HEIGHT * 0.1)

    line_spacing = int(HEIGHT * 0.05)

    # Stats
    for i, (key, value) in enumerate(metrics.items()):
        stat_text = FONT.render(
            f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}",
            True,
            BLACK,
        )
        screen.blit(stat_text, (stats_x, stats_y + line_spacing * i))

    # Back button
    back_button_rect = pygame.Rect(
        WIDTH // 2 - button_width // 2,
        HEIGHT - button_height - line_spacing,
        button_width,
        button_height,
    )

    pygame.draw.rect(screen, BLUE, back_button_rect)
    pygame.draw.rect(screen, BLACK, back_button_rect, 2)

    back_button = FONT.render("Back", True, WHITE)
    screen.blit(
        back_button,
        (
            back_button_rect.x + button_width // 2 - back_button.get_width() // 2,
            back_button_rect.y + button_height // 2 - back_button.get_height() // 2,
        ),
    )

    return back_button_rect


def _initialize_high_demand_zone(grid_size: int) -> Tuple[int, int, int]:
    zone_center_x = random.randint(0, grid_size - 1)
    zone_center_y = random.randint(0, grid_size - 1)
    zone_radius = max(1, grid_size // 4)

    return zone_center_x, zone_center_y, zone_radius


def _is_within_high_demand_zone(
    x: int, y: int, center_x: int, center_y: int, radius: int
) -> bool:
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def _initialize_vehicles(
    init_strategy: str, high_demand_x, high_demand_y, high_demand_radius
) -> List[Vehicle]:
    if init_strategy == "center":
        return [Vehicle(GRID_SIZE // 2, GRID_SIZE // 2) for _ in range(TOTAL_VEHICLES)]

    elif init_strategy == "uniform":
        step = max(1, GRID_SIZE // int(TOTAL_VEHICLES**0.5))
        vehicles = [
            Vehicle(x, y)
            for x in range(0, GRID_SIZE, step)
            for y in range(0, GRID_SIZE, step)
        ]

        # Ensure correct number of vehicles
        while len(vehicles) < TOTAL_VEHICLES:
            vehicles.append(
                Vehicle(
                    random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                )
            )
        return vehicles[:TOTAL_VEHICLES]

    elif init_strategy == "high_demand" and HIGH_DEMAND_ZONE:
        return [
            Vehicle(
                random.randint(
                    max(0, high_demand_x - high_demand_radius),
                    min(GRID_SIZE - 1, high_demand_x + high_demand_radius),
                ),
                random.randint(
                    max(0, high_demand_y - high_demand_radius),
                    min(GRID_SIZE - 1, high_demand_y + high_demand_radius),
                ),
            )
            for _ in range(TOTAL_VEHICLES)
        ]

    elif init_strategy == "random":
        return [
            Vehicle(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            for _ in range(TOTAL_VEHICLES)
        ]

    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")


def _draw_vehicle(screen: pygame.Surface, vehicle: Vehicle):
    color = (
        GREEN
        if vehicle.state == State.AVAILABLE
        else RED if vehicle.state == State.NEEDS_RECHARGING else BLUE
    )
    pygame.draw.rect(
        screen,
        color,
        pygame.Rect(
            vehicle.x * CELL_WIDTH,
            vehicle.y * CELL_HEIGHT,
            CELL_WIDTH,
            CELL_HEIGHT,
        ),
    )
    if vehicle.user:
        pygame.draw.line(
            screen,
            BLACK,
            (
                vehicle.x * CELL_WIDTH + CELL_WIDTH // 2,
                vehicle.y * CELL_HEIGHT + CELL_HEIGHT // 2,
            ),
            (
                vehicle.user.goal_x * CELL_WIDTH + CELL_WIDTH // 2,
                vehicle.user.goal_y * CELL_HEIGHT + CELL_HEIGHT // 2,
            ),
        )


async def run_simulation(
    width: int,
    height: int,
    grid_size: int,
    total_vehicles: int,
    battery_depletion_rate: int,
    recharge_time: int,
    max_users: int,
    user_probability: float,
    fps: int,
    high_demand_zone: bool,
    vehicle_init: str,
):
    global FONT, FILE, WIDTH, HEIGHT, GRID_SIZE, CELL_HEIGHT, CELL_WIDTH, TOTAL_VEHICLES, BATTERY_DEPLETION_RATE, RECHARGE_TIME, MAX_USERS, USER_PROBABILITY, FPS, HIGH_DEMAND_ZONE, VEHICLE_INIT
    _set_global_parameters(
        width,
        height,
        grid_size,
        total_vehicles,
        battery_depletion_rate,
        recharge_time,
        max_users,
        user_probability,
        fps,
        high_demand_zone,
        vehicle_init,
    )

    # Init pygame
    pygame.init()
    pygame.display.set_caption("MAS - Project 3 (Nikethan & Luca)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game_screen_rect = pygame.Rect(10, 10, WIDTH - 20, HEIGHT - 20)

    # High demand zone
    high_demand_x, high_demand_y, high_demand_radius = None, None, None
    if HIGH_DEMAND_ZONE:
        high_demand_x, high_demand_y, high_demand_radius = _initialize_high_demand_zone(
            GRID_SIZE
        )

    # Init entities
    vehicles: List[Vehicle] = _initialize_vehicles(
        VEHICLE_INIT, high_demand_x, high_demand_y, high_demand_radius
    )
    users: List[User] = []

    # Run simulation
    total_frames: int = 0
    running = True
    while running:
        await asyncio.sleep(0)
        screen.fill(WHITE)

        # Quit or return to menu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos):
                    running = False

        # Generate users
        if len(users) < MAX_USERS:
            spawn_prob = USER_PROBABILITY
            user_x, user_y = random.randint(0, GRID_SIZE - 1), random.randint(
                0, GRID_SIZE - 1
            )

            if HIGH_DEMAND_ZONE and _is_within_high_demand_zone(
                user_x, user_y, high_demand_x, high_demand_y, high_demand_radius
            ):
                spawn_prob *= 100

            if random.random() < spawn_prob:
                users.append(User(user_x, user_y))

        # Update users
        users.sort(key=lambda user: user.wait_time, reverse=True)
        for user in users:
            # Delete user if reached goal
            if user.has_reached_goal():
                if user.vehicle:
                    if user.vehicle.battery > 0:
                        user.vehicle.state = State.AVAILABLE
                    else:
                        user.vehicle.state = State.NEEDS_RECHARGING
                    user.vehicle.user = None
                    user.vehicle = None

                users.remove(user)
                continue

            # Check for vehicles (rent)
            if user.vehicle is None:
                for vehicle in vehicles:
                    if (
                        vehicle.state == State.AVAILABLE
                        and abs(vehicle.x - user.x) + abs(vehicle.y - user.y) <= 1
                    ):
                        user.vehicle = vehicle
                        vehicle.user = user
                        vehicle.state = State.IN_USE
                        break
            # Update vehicle in use
            else:
                user.vehicle.x = user.x
                user.vehicle.y = user.y

                user.vehicle.battery -= BATTERY_DEPLETION_RATE
                if user.vehicle.battery <= 0:
                    user.vehicle.state = State.NEEDS_RECHARGING
                    user.vehicle.user = None
                    user.vehicle = None

            pygame.draw.circle(
                screen,
                BLACK,
                (
                    user.x * CELL_WIDTH + CELL_WIDTH // 2,
                    user.y * CELL_HEIGHT + CELL_HEIGHT // 2,
                ),
                min(CELL_WIDTH, CELL_HEIGHT) // 4,
            )

            user.update()

        # Update vehicles
        for vehicle in vehicles:
            _draw_vehicle(screen, vehicle)

            vehicle.update()

        # Draw grid
        for i in range(GRID_SIZE):
            pygame.draw.line(
                screen, LIGHT_GRAY, (0, i * CELL_HEIGHT), (WIDTH, i * CELL_HEIGHT)
            )
            pygame.draw.line(
                screen, LIGHT_GRAY, (i * CELL_WIDTH, 0), (i * CELL_WIDTH, HEIGHT)
            )

        if HIGH_DEMAND_ZONE:
            pygame.draw.circle(
                screen,
                LIGHT_GRAY,
                (
                    high_demand_x * CELL_WIDTH + CELL_WIDTH // 2,
                    high_demand_y * CELL_HEIGHT + CELL_HEIGHT // 2,
                ),
                high_demand_radius * CELL_WIDTH,
                2,
            )

        # Calculate and write metrics
        metrics = _calculate_metrics(vehicles, users, total_frames)
        _write_metrics(metrics)

        # Render stats and back button
        if game_screen_rect.collidepoint(pygame.mouse.get_pos()):
            back_button_rect = _render_stats_and_back(screen, metrics)

        if total_frames % FPS == 0:
            logger.info(f"Frame: {total_frames}")

        pygame.display.flip()
        clock.tick(FPS)
        total_frames += 1
