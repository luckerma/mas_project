from asyncio import run, sleep
from pathlib import Path
from sys import exit as sys_exit
from typing import Dict, Tuple

import pygame

from colors import BLACK, BLUE, GREEN, LIGHT_GRAY, WHITE
from simulation import run_simulation

ROOT = Path(__file__).parent
FAVICON_FILE = ROOT / "favicon.png"


class ConfigMenu:
    """Configuration menu for setting simulation parameters."""

    def __init__(self):
        self.params = {
            "WIDTH": 800,
            "HEIGHT": 600,
            "GRID_SIZE": 12,
            "TOTAL_VEHICLES": 10,
            "BATTERY_DEPLETION_RATE": 20,
            "RECHARGE_TIME": 30,
            "MAX_USERS": 10,
            "USER_PROBABILITY": 0.2,
            "FPS": 30,
            "HIGH_DEMAND_ZONE": False,
            "VEHICLE_INIT": "random",
        }

        self.slider_ranges = {
            "WIDTH": (400, 1200),
            "HEIGHT": (300, 900),
            "GRID_SIZE": (4, 16),
            "TOTAL_VEHICLES": (1, 100),
            "BATTERY_DEPLETION_RATE": (1, 100),
            "RECHARGE_TIME": (10, 300),
            "MAX_USERS": (1, 100),
            "USER_PROBABILITY": (0.1, 1.0),
            "FPS": (1, 60),
        }

        self.integer_params = {
            "WIDTH",
            "HEIGHT",
            "GRID_SIZE",
            "TOTAL_VEHICLES",
            "BATTERY_DEPLETION_RATE",
            "RECHARGE_TIME",
            "MAX_USERS",
            "FPS",
        }

        self.vehicle_init_options = [
            "random",
            "center",
            "uniform",
            "high_demand",
        ]

        self.font = None
        self.screen = None
        self.selected_param = None
        self.dropdown_selected_index = self.vehicle_init_options.index(
            self.params.get("VEHICLE_INIT", "random")
        )
        self.sliders: Dict[str, Tuple[float, pygame.Rect, pygame.Rect]] = {}

    def draw_slider(
        self,
        param: str,
        value: float,
        x: int,
        y: int,
        min_val: float,
        max_val: float,
        width: int,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        # Slider background
        slider_rect = pygame.Rect(x, y, width, 10)
        pygame.draw.rect(self.screen, LIGHT_GRAY, slider_rect)
        pygame.draw.rect(self.screen, BLACK, slider_rect, 2)

        # Handle position
        slider_pos = int((value - min_val) / (max_val - min_val) * width)
        handle_rect = pygame.Rect(x + slider_pos - 5, y - 5, 10, 20)
        pygame.draw.rect(self.screen, GREEN, handle_rect)

        # Parameter label
        label = self.font.render(
            f"{param.replace('_', ' ')}: {int(value) if param in self.integer_params else value:.2f}",
            True,
            BLACK,
        )
        self.screen.blit(label, (x, y - 30))

        return slider_rect, handle_rect

    def draw_menu(self) -> Tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        self.screen.fill(WHITE)

        # Dynamic spacing
        padding: int = int(self.params["WIDTH"] * 0.05)
        spacing: int = int(self.params["HEIGHT"] * 0.08)
        slider_width: int = int(self.params["WIDTH"] * 0.8)

        # Draw sliders
        for i, (param, (value, _, _)) in enumerate(self.sliders.items()):
            min_val, max_val = self.slider_ranges[param]
            rect, handle = self.draw_slider(
                param,
                value,
                int(self.params["WIDTH"] * 0.1),
                padding + i * spacing,
                min_val,
                max_val,
                slider_width,
            )
            self.sliders[param] = (value, rect, handle)

        # Checkbox (High Demand Zone)
        checkbox_size = int(self.params["WIDTH"] * 0.03)
        checkbox_y_offset = padding + len(self.sliders) * spacing
        checkbox_rect = pygame.Rect(
            int(self.params["WIDTH"] * 0.1),
            checkbox_y_offset,
            checkbox_size,
            checkbox_size,
        )
        pygame.draw.rect(self.screen, LIGHT_GRAY, checkbox_rect)
        if self.params["HIGH_DEMAND_ZONE"]:
            pygame.draw.line(
                self.screen,
                GREEN,
                (checkbox_rect.left, checkbox_rect.top),
                (checkbox_rect.right, checkbox_rect.bottom),
                2,
            )
            pygame.draw.line(
                self.screen,
                GREEN,
                (checkbox_rect.left, checkbox_rect.bottom),
                (checkbox_rect.right, checkbox_rect.top),
                2,
            )
        pygame.draw.rect(self.screen, BLACK, checkbox_rect, 2)
        checkbox_label = self.font.render("HIGH DEMAND ZONE", True, BLACK)
        self.screen.blit(checkbox_label, (checkbox_rect.right + 10, checkbox_rect.top))

        # Dropdown (Vehicle Init)
        dropdown_width = int(self.params["WIDTH"] * 0.4)
        dropdown_height = int(self.params["HEIGHT"] * 0.05)
        dropdown_rect = pygame.Rect(
            int(self.params["WIDTH"] * 0.5),
            checkbox_y_offset,
            dropdown_width,
            dropdown_height,
        )
        pygame.draw.rect(self.screen, LIGHT_GRAY, dropdown_rect)
        pygame.draw.rect(self.screen, BLACK, dropdown_rect, 2)
        dropdown_label = self.font.render(
            f"VEHICLE INIT: {self.params['VEHICLE_INIT']}", True, BLACK
        )
        self.screen.blit(
            dropdown_label,
            (
                dropdown_rect.x + (dropdown_width - dropdown_label.get_width()) // 2,
                dropdown_rect.y + (dropdown_height - dropdown_label.get_height()) // 2,
            ),
        )

        # Start button
        button_width = int(self.params["WIDTH"] * 0.25)
        button_height = int(self.params["HEIGHT"] * 0.1)
        start_button_rect = pygame.Rect(
            self.params["WIDTH"] // 2 - button_width // 2,
            self.params["HEIGHT"] - padding // 2 - button_height,
            button_width,
            button_height,
        )
        pygame.draw.rect(self.screen, BLUE, start_button_rect)
        pygame.draw.rect(self.screen, BLACK, start_button_rect, 2)
        start_button = self.font.render("Start", True, WHITE)
        self.screen.blit(
            start_button,
            (
                start_button_rect.x + button_width // 2 - start_button.get_width() // 2,
                start_button_rect.y
                + button_height // 2
                - start_button.get_height() // 2,
            ),
        )

        return checkbox_rect, dropdown_rect, start_button_rect

    async def run(self):
        """Runs the configuration menu loop."""

        # Pygame setup
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Configuration Menu")
        if FAVICON_FILE.exists():
            pygame.display.set_icon(pygame.image.load(str(FAVICON_FILE)))
        self.screen = pygame.display.set_mode(
            (self.params["WIDTH"], self.params["HEIGHT"])
        )
        self.font = pygame.font.Font(None, int(self.params["WIDTH"] * 0.04))

        # Initialize sliders
        for param in self.slider_ranges.keys():
            value = self.params[param]
            self.sliders[param] = (value, None, None)

        running = True
        while running:
            await sleep(0)
            checkbox_rect, dropdown_rect, start_button_rect = self.draw_menu()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Slider selection
                    for param, (value, _, handle_rect) in self.sliders.items():
                        if handle_rect.collidepoint(event.pos):
                            self.selected_param = param
                            break

                    # Checkbox toggle
                    if checkbox_rect.collidepoint(event.pos):
                        self.params["HIGH_DEMAND_ZONE"] = not self.params[
                            "HIGH_DEMAND_ZONE"
                        ]

                    # Dropdown selection
                    if dropdown_rect.collidepoint(event.pos):
                        self.dropdown_selected_index = (
                            self.dropdown_selected_index + 1
                        ) % len(self.vehicle_init_options)
                        self.params["VEHICLE_INIT"] = self.vehicle_init_options[
                            self.dropdown_selected_index
                        ]

                    # Start button
                    if start_button_rect.collidepoint(event.pos):
                        # Ensure high demand zone is enabled if vehicle init is set to high demand
                        if self.params["VEHICLE_INIT"] == "high_demand":
                            self.params["HIGH_DEMAND_ZONE"] = True

                        # Update parameters with slider values
                        for param, (value, _, _) in self.sliders.items():
                            self.params[param] = (
                                int(value) if param in self.integer_params else value
                            )

                        # Start simulation
                        await run_simulation(
                            width=self.params["WIDTH"],
                            height=self.params["HEIGHT"],
                            grid_size=self.params["GRID_SIZE"],
                            total_vehicles=self.params["TOTAL_VEHICLES"],
                            battery_depletion_rate=self.params[
                                "BATTERY_DEPLETION_RATE"
                            ],
                            recharge_time=self.params["RECHARGE_TIME"],
                            max_users=self.params["MAX_USERS"],
                            user_probability=self.params["USER_PROBABILITY"],
                            fps=self.params["FPS"],
                            high_demand_zone=self.params["HIGH_DEMAND_ZONE"],
                            vehicle_init=self.params["VEHICLE_INIT"],
                        )

                        # Reset the menu screen after simulation
                        pygame.display.set_caption("Configuration Menu")
                        self.screen = pygame.display.set_mode(
                            (self.params["WIDTH"], self.params["HEIGHT"])
                        )
                        self.font = pygame.font.Font(
                            None, int(self.params["WIDTH"] * 0.04)
                        )

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.selected_param = None

                # Slider adjustment
                elif event.type == pygame.MOUSEMOTION and self.selected_param:
                    min_val, max_val = self.slider_ranges[self.selected_param]
                    x, y, width = (
                        self.sliders[self.selected_param][1].x,
                        self.sliders[self.selected_param][1].y,
                        self.sliders[self.selected_param][1].width,
                    )
                    rel_x = max(0, min(event.pos[0] - x, width))
                    new_value = min_val + (rel_x / width) * (max_val - min_val)

                    if self.selected_param in self.integer_params:
                        new_value = round(new_value)

                    self.sliders[self.selected_param] = (
                        new_value,
                        self.sliders[self.selected_param][1],
                        self.sliders[self.selected_param][2],
                    )

        pygame.quit()
        sys_exit()


async def main():
    """Main function to run the configuration menu."""

    config_menu = ConfigMenu()
    await config_menu.run()


if __name__ == "__main__":
    run(main())
