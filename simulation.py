from asyncio import sleep
from csv import DictWriter
from datetime import datetime
from enum import Enum
from logging import Logger
from pathlib import Path
from random import randint, random
from typing import Dict, List, Optional, Tuple

import pygame

from colors import BLACK, BLUE, GREEN, LIGHT_GRAY, RED, WHITE

logger = Logger(__name__)
ROOT_DIR: Path = Path(__file__).parent


class Config:
    """Configuration parameters for the simulation."""

    def __init__(
        self,
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
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.total_vehicles = total_vehicles
        self.battery_depletion_rate = battery_depletion_rate
        self.recharge_time = recharge_time
        self.max_users = max_users
        self.user_probability = user_probability
        self.fps = fps
        self.high_demand_zone = high_demand_zone
        self.vehicle_init = vehicle_init

        self.cell_height = self.height / self.grid_size
        self.cell_width = self.width / self.grid_size
        self.font_size = int(self.width * 0.04)
        self.font = pygame.font.Font(None, self.font_size)
        self.file = Path(
            ROOT_DIR / f"metrics/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        if not self.file.parent.exists():
            self.file.parent.mkdir(parents=True, exist_ok=True)


### Entities ###


class State(Enum):
    """Represents the state of a vehicle agent."""

    AVAILABLE = "Available"
    IN_USE = "In Use"
    NEEDS_RECHARGING = "Needs Recharging"


class Vehicle:
    """Represents a vehicle agent in the simulation."""

    def __init__(self, x: int, y: int, config: Config):
        self.x = x
        self.y = y
        self.config = config

        self.battery: int = 100
        self.state: State = State.AVAILABLE
        self.recharge_timer: int = 0

    def is_available(self) -> bool:
        return self.state == State.AVAILABLE

    def use(self):
        if self.is_available():
            self.state = State.IN_USE

    def deplete_battery(self):
        self.battery -= self.config.battery_depletion_rate
        if self.battery <= 0:
            self.battery = 0
            self.state = State.NEEDS_RECHARGING

    def recharge(self):
        if self.state == State.NEEDS_RECHARGING:
            self.recharge_timer += 1
            if self.recharge_timer >= self.config.recharge_time:
                self.battery = 100
                self.state = State.AVAILABLE
                self.recharge_timer = 0

    def update(self):
        if self.state == State.NEEDS_RECHARGING:
            self.recharge()


class User:
    """Represents a user agent in the simulation."""

    def __init__(self, x: int, y: int, config: Config):
        self.x = x
        self.y = y
        self.config = config

        self.goal_x, self.goal_y = self._generate_goal()
        self.wait_time: int = 0
        self.assigned_vehicle: Vehicle = None

    def _generate_goal(self) -> Tuple[int, int]:
        while True:
            goal_x = randint(0, self.config.grid_size - 1)
            goal_y = randint(0, self.config.grid_size - 1)
            if goal_x != self.x or goal_y != self.y:
                return goal_x, goal_y

    def is_near(self, vehicle: Vehicle) -> bool:
        return abs(vehicle.x - self.x) + abs(vehicle.y - self.y) <= 1

    def has_reached_goal(self) -> bool:
        return self.x == self.goal_x and self.y == self.goal_y

    def move_towards_goal(self):
        if self.x < self.goal_x:
            self.x += 1
        elif self.x > self.goal_x:
            self.x -= 1

        if self.y < self.goal_y:
            self.y += 1
        elif self.y > self.goal_y:
            self.y -= 1

    def find_vehicle(self, vehicles: List[Vehicle]) -> Optional[Vehicle]:
        for vehicle in vehicles:
            if vehicle.is_available() and self.is_near(vehicle):
                return vehicle

        return None

    def rent_vehicle(self, vehicle: Vehicle):
        if vehicle.is_available():
            vehicle.use()
            self.assigned_vehicle = vehicle

    def release_vehicle(self):
        if self.assigned_vehicle:
            if self.assigned_vehicle.battery > 0:
                self.assigned_vehicle.state = State.AVAILABLE
            else:
                self.assigned_vehicle.state = State.NEEDS_RECHARGING
            self.assigned_vehicle = None

    def update(self):
        if self.assigned_vehicle:
            self.assigned_vehicle.x = self.x
            self.assigned_vehicle.y = self.y
            self.assigned_vehicle.deplete_battery()
            if self.assigned_vehicle.battery <= 0:
                self.release_vehicle()
        else:
            self.wait_time += 1
        self.move_towards_goal()


class Environment:
    """Represents the simulation environment."""

    def __init__(self, config: Config):
        self.config = config

        self.vehicles: List[Vehicle] = []
        self.users: List[User] = []
        self.high_demand_x: int = None
        self.high_demand_y: int = None
        self.high_demand_radius: int = None

        self._initialize_high_demand_zone()
        self._initialize_vehicles()

    def _initialize_high_demand_zone(self):
        if self.config.high_demand_zone:
            self.high_demand_x = randint(0, self.config.grid_size - 1)
            self.high_demand_y = randint(0, self.config.grid_size - 1)
            self.high_demand_radius = max(1, self.config.grid_size // 4)

    def _is_within_high_demand_zone(self, x: int, y: int) -> bool:
        if not self.config.high_demand_zone:
            return False

        return (x - self.high_demand_x) ** 2 + (
            y - self.high_demand_y
        ) ** 2 <= self.high_demand_radius**2

    def _initialize_vehicles(self):
        if self.config.vehicle_init == "center":
            self.vehicles = [
                Vehicle(
                    self.config.grid_size // 2, self.config.grid_size // 2, self.config
                )
                for _ in range(self.config.total_vehicles)
            ]
        elif self.config.vehicle_init == "uniform":
            step = max(1, self.config.grid_size // int(self.config.total_vehicles**0.5))
            self.vehicles = [
                Vehicle(x, y, self.config)
                for x in range(0, self.config.grid_size, step)
                for y in range(0, self.config.grid_size, step)
            ][: self.config.total_vehicles]
            while len(self.vehicles) < self.config.total_vehicles:
                self.vehicles.append(
                    Vehicle(
                        randint(0, self.config.grid_size - 1),
                        randint(0, self.config.grid_size - 1),
                        self.config,
                    )
                )
        elif self.config.vehicle_init == "high_demand" and self.config.high_demand_zone:
            self.vehicles = [
                Vehicle(
                    randint(
                        max(0, self.high_demand_x - self.high_demand_radius),
                        min(
                            self.config.grid_size - 1,
                            self.high_demand_x + self.high_demand_radius,
                        ),
                    ),
                    randint(
                        max(0, self.high_demand_y - self.high_demand_radius),
                        min(
                            self.config.grid_size - 1,
                            self.high_demand_y + self.high_demand_radius,
                        ),
                    ),
                    self.config,
                )
                for _ in range(self.config.total_vehicles)
            ]
        elif self.config.vehicle_init == "random":
            self.vehicles = [
                Vehicle(
                    randint(0, self.config.grid_size - 1),
                    randint(0, self.config.grid_size - 1),
                    self.config,
                )
                for _ in range(self.config.total_vehicles)
            ]
        else:
            raise ValueError(
                f"Unknown initialization strategy: {self.config.vehicle_init}"
            )

    def spawn_users(self):
        if len(self.users) >= self.config.max_users:
            return

        spawn_prob = self.config.user_probability
        user_x = randint(0, self.config.grid_size - 1)
        user_y = randint(0, self.config.grid_size - 1)

        if self._is_within_high_demand_zone(user_x, user_y):
            spawn_prob *= 100

        if random() < spawn_prob:
            self.users.append(User(user_x, user_y, self.config))

    def match_users_to_vehicles(self):
        for user in self.users:
            if user.assigned_vehicle is None:
                vehicle = user.find_vehicle(self.vehicles)
                if vehicle:
                    user.rent_vehicle(vehicle)

    def update_agents(self):
        for user in self.users[:]:
            user.update()
            if user.has_reached_goal():
                user.release_vehicle()
                self.users.remove(user)
        for vehicle in self.vehicles:
            vehicle.update()

    def update(self):
        self.spawn_users()
        self.match_users_to_vehicles()
        self.update_agents()


### Metrics ###


def calculate_metrics(environment: Environment, frame: int) -> Dict[str, float]:
    """
    Calculate simulation metrics.

    Args:
        environment (Environment): The simulation environment.
        frame (int): The current frame number.

    Returns:
        Dict[str, float]: A dictionary containing metric names and their values.
    """

    vehicles = environment.vehicles
    users = environment.users

    avg_wait_time = sum(user.wait_time for user in users) / len(users) if users else 0
    available_percentage = (
        sum(1 for vehicle in vehicles if vehicle.state == State.AVAILABLE)
        / len(vehicles)
        * 100
    )
    vehicles_depleted = sum(
        1 for vehicle in vehicles if vehicle.state == State.NEEDS_RECHARGING
    )
    utilization_percentage = (
        sum(1 for vehicle in vehicles if vehicle.state == State.IN_USE)
        / (len(vehicles) - vehicles_depleted)
        if (len(vehicles) - vehicles_depleted) > 0
        else 0
    )

    avg_wait_time = round(avg_wait_time, 2)
    available_percentage = round(available_percentage, 2)
    utilization_percentage = round(utilization_percentage, 2)

    return {
        "frame": frame,
        "users": len(users),
        "vehicles": len(vehicles),
        "average_user_wait_time": avg_wait_time,
        "vehicle_availability": available_percentage,
        "vehicles_depleted": vehicles_depleted,
        "vehicle_utilization": utilization_percentage,
    }


def write_metrics(metrics: Dict[str, float], config: Config):
    """Write simulation metrics to the CSV file."""

    with open(config.file, "a", newline="") as f:
        writer = DictWriter(f, fieldnames=metrics.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)


class Renderer:
    """Handles rendering of the simulation."""

    def __init__(
        self, screen: pygame.Surface, environment: Environment, config: Config
    ):
        self.screen = screen
        self.environment = environment
        self.config = config

        self.game_screen_rect = pygame.Rect(
            10, 10, self.config.width - 20, self.config.height - 20
        )

    def draw_grid(self):
        for i in range(self.config.grid_size):
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (0, i * self.config.cell_height),
                (self.config.width, i * self.config.cell_height),
            )
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (i * self.config.cell_width, 0),
                (i * self.config.cell_width, self.config.height),
            )

    def draw_high_demand_zone(self):
        if self.config.high_demand_zone:
            pygame.draw.circle(
                self.screen,
                LIGHT_GRAY,
                (
                    self.environment.high_demand_x * self.config.cell_width
                    + self.config.cell_width // 2,
                    self.environment.high_demand_y * self.config.cell_height
                    + self.config.cell_height // 2,
                ),
                self.environment.high_demand_radius * self.config.cell_width,
                2,
            )

    def draw_vehicle(self, vehicle: Vehicle):
        color = (
            GREEN
            if vehicle.state == State.AVAILABLE
            else RED if vehicle.state == State.NEEDS_RECHARGING else BLUE
        )
        user = next(
            (
                user
                for user in self.environment.users
                if user.assigned_vehicle == vehicle
            ),
            None,
        )
        if user:
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    user.x * self.config.cell_width,
                    user.y * self.config.cell_height,
                    self.config.cell_width,
                    self.config.cell_height,
                ),
            )
        else:
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    vehicle.x * self.config.cell_width,
                    vehicle.y * self.config.cell_height,
                    self.config.cell_width,
                    self.config.cell_height,
                ),
            )

    def draw_user(self, user: User):
        pygame.draw.circle(
            self.screen,
            BLACK,
            (
                user.x * self.config.cell_width + self.config.cell_width // 2,
                user.y * self.config.cell_height + self.config.cell_height // 2,
            ),
            min(self.config.cell_width, self.config.cell_height) // 4,
        )
        if user.assigned_vehicle:
            pygame.draw.line(
                self.screen,
                BLACK,
                (
                    user.x * self.config.cell_width + self.config.cell_width // 2,
                    user.y * self.config.cell_height + self.config.cell_height // 2,
                ),
                (
                    user.goal_x * self.config.cell_width + self.config.cell_width // 2,
                    user.goal_y * self.config.cell_height
                    + self.config.cell_height // 2,
                ),
            )

    def render_stats_and_back(self, metrics: Dict[str, float]) -> pygame.Rect:
        stats_x = int(self.config.width * 0.02)
        stats_y = int(self.config.height * 0.02)

        button_width = int(self.config.width * 0.25)
        button_height = int(self.config.height * 0.1)

        line_spacing = int(self.config.height * 0.05)

        # Stats
        for i, (key, value) in enumerate(metrics.items()):
            stat_text = self.config.font.render(
                (
                    f"{key}: {value:.2f}"
                    if isinstance(value, float)
                    else f"{key}: {value}"
                ),
                True,
                BLACK,
            )
            self.screen.blit(stat_text, (stats_x, stats_y + line_spacing * i))

        # Back button
        back_button_rect = pygame.Rect(
            self.config.width // 2 - button_width // 2,
            self.config.height - button_height - line_spacing,
            button_width,
            button_height,
        )

        pygame.draw.rect(self.screen, BLUE, back_button_rect)
        pygame.draw.rect(self.screen, BLACK, back_button_rect, 2)

        back_button = self.config.font.render("Back", True, WHITE)
        self.screen.blit(
            back_button,
            (
                back_button_rect.x + button_width // 2 - back_button.get_width() // 2,
                back_button_rect.y + button_height // 2 - back_button.get_height() // 2,
            ),
        )

        return back_button_rect

    def draw(self, metrics: Dict[str, float]) -> Optional[pygame.Rect]:
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_high_demand_zone()

        for vehicle in self.environment.vehicles:
            self.draw_vehicle(vehicle)
        for user in self.environment.users:
            self.draw_user(user)

        # Only render if mouse is inside window
        mouse_pos = pygame.mouse.get_pos()
        if self.game_screen_rect.collidepoint(mouse_pos):
            back_button_rect = self.render_stats_and_back(metrics)
        else:
            back_button_rect = None

        pygame.display.flip()

        return back_button_rect


### Simulation ###


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
    """
    Run the multi-agent system simulation.

    Args:
        width (int): Screen width.
        height (int): Screen height.
        grid_size (int): Size of the grid.
        total_vehicles (int): Total number of vehicles.
        battery_depletion_rate (int): Battery depletion rate per update.
        recharge_time (int): Time required for a vehicle to recharge.
        max_users (int): Maximum number of users.
        user_probability (float): Probability of a new user appearing.
        fps (int): Frames per second.
        high_demand_zone (bool): Enable high demand zone.
        vehicle_init (str): Vehicle initialization strategy.
    """

    config = Config(
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

    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("MAS Simulation for Shared Micromobility")
    screen = pygame.display.set_mode((config.width, config.height))
    clock = pygame.time.Clock()

    # Initialize environment and renderer
    environment = Environment(config)
    renderer = Renderer(screen, environment, config)

    # Run simulation
    total_frames: int = 0
    running = True
    while running:
        await sleep(0)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect and back_button_rect.collidepoint(event.pos):
                    running = False

        # Update environment
        environment.update()

        # Calculate and write metrics
        metrics = calculate_metrics(environment, total_frames)
        write_metrics(metrics, config)

        # Draw the simulation
        back_button_rect = renderer.draw(metrics)

        if total_frames % config.fps == 0:
            logger.info(f"Frame: {total_frames}")

        clock.tick(config.fps)
        total_frames += 1
