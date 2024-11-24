import asyncio
import sys
from pathlib import Path
from typing import Dict, Tuple

import pygame

from colors import BLACK, BLUE, GREEN, LIGHT_GRAY, WHITE
from simulation import run_simulation

ROOT: Path = Path(__file__).parent
FAVICON_FILE: Path = ROOT / "favicon.png"

FONT: pygame.font.Font

# Default parameters and ranges
params = {
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

slider_ranges = {
    "WIDTH": (400, 1200),
    "HEIGHT": (300, 900),
    "GRID_SIZE": (4, 16),
    "TOTAL_VEHICLES": (1, 10),
    "BATTERY_DEPLETION_RATE": (1, 100),
    "RECHARGE_TIME": (10, 300),
    "MAX_USERS": (1, 10),
    "USER_PROBABILITY": (0.1, 1.0),
    "FPS": (1, 30),
}

integer_params = {
    "WIDTH",
    "HEIGHT",
    "GRID_SIZE",
    "TOTAL_VEHICLES",
    "BATTERY_DEPLETION_RATE",
    "RECHARGE_TIME",
    "MAX_USERS",
    "FPS",
}

vehicle_init_options = [
    "random",
    "center",
    "uniform",
    "high_demand",
]


def _draw_slider(
    screen: pygame.Surface,
    param: str,
    value: float,
    x: int,
    y: int,
    min_val: float,
    max_val: float,
    width: int,
) -> Tuple[pygame.Rect, pygame.Rect]:
    # Slider
    slider_rect = pygame.Rect(x, y, width, 10)
    pygame.draw.rect(screen, LIGHT_GRAY, slider_rect)
    pygame.draw.rect(screen, BLACK, slider_rect, 2)

    # Position
    slider_pos = int((value - min_val) / (max_val - min_val) * width)
    handle_rect = pygame.Rect(x + slider_pos - 5, y - 5, 10, 20)
    pygame.draw.rect(screen, GREEN, handle_rect)

    # Param and value
    label = FONT.render(
        f"{param.replace('_', ' ')}: {int(value) if param in integer_params else value:.2f}",
        True,
        BLACK,
    )
    screen.blit(label, (x, y - 30))

    return slider_rect, handle_rect


def _draw_menu(
    screen: pygame.Surface, sliders: Dict
) -> Tuple[Dict, pygame.Rect, pygame.Rect, pygame.Rect]:
    screen.fill(WHITE)

    # Dynamic spacing
    padding: int = int(params["WIDTH"] * 0.05)
    spacing: int = int(params["HEIGHT"] * 0.08)
    slider_width: int = int(params["WIDTH"] * 0.8)

    # Sliders
    for i, (param, (value, _, _)) in enumerate(sliders.items()):
        min_val, max_val = slider_ranges[param]
        rect, handle = _draw_slider(
            screen,
            param,
            value,
            int(params["WIDTH"] * 0.1),
            padding + i * spacing,
            min_val,
            max_val,
            slider_width,
        )
        sliders[param] = (value, rect, handle)

    # Checkbox (High Demand Zone)
    checkbox_size = int(params["WIDTH"] * 0.03)
    checkbox_y_offset = padding + len(sliders) * spacing
    checkbox_rect = pygame.Rect(
        int(params["WIDTH"] * 0.1),
        checkbox_y_offset,
        checkbox_size,
        checkbox_size,
    )
    pygame.draw.rect(screen, LIGHT_GRAY, checkbox_rect)
    if params["HIGH_DEMAND_ZONE"]:
        pygame.draw.line(
            screen,
            GREEN,
            (checkbox_rect.left, checkbox_rect.top),
            (checkbox_rect.right, checkbox_rect.bottom),
            2,
        )
        pygame.draw.line(
            screen,
            GREEN,
            (checkbox_rect.left, checkbox_rect.bottom),
            (checkbox_rect.right, checkbox_rect.top),
            2,
        )
    pygame.draw.rect(screen, BLACK, checkbox_rect, 2)
    checkbox_label = FONT.render("HIGH DEMAND ZONE", True, BLACK)
    screen.blit(checkbox_label, (checkbox_rect.right + 10, checkbox_rect.top))

    # Dropdown (Vehicle Init)
    dropdown_width = int(params["WIDTH"] * 0.4)
    dropdown_height = int(params["HEIGHT"] * 0.05)
    dropdown_rect = pygame.Rect(
        int(params["WIDTH"] * 0.5),
        checkbox_y_offset,
        dropdown_width,
        dropdown_height,
    )
    pygame.draw.rect(screen, LIGHT_GRAY, dropdown_rect)
    pygame.draw.rect(screen, BLACK, dropdown_rect, 2)
    dropdown_label = FONT.render(f"VEHICLE INIT: {params['VEHICLE_INIT']}", True, BLACK)
    screen.blit(
        dropdown_label,
        (
            dropdown_rect.x + (dropdown_width - dropdown_label.get_width()) // 2,
            dropdown_rect.y + (dropdown_height - dropdown_label.get_height()) // 2,
        ),
    )

    # Start button
    button_width = int(params["WIDTH"] * 0.25)
    button_height = int(params["HEIGHT"] * 0.1)
    start_button_rect = pygame.Rect(
        params["WIDTH"] // 2 - button_width // 2,
        params["HEIGHT"] - padding // 2 - button_height,
        button_width,
        button_height,
    )
    pygame.draw.rect(screen, BLUE, start_button_rect)
    pygame.draw.rect(screen, BLACK, start_button_rect, 2)
    start_button = FONT.render("Start", True, WHITE)
    screen.blit(
        start_button,
        (
            start_button_rect.x + button_width // 2 - start_button.get_width() // 2,
            start_button_rect.y + button_height // 2 - start_button.get_height() // 2,
        ),
    )

    return sliders, checkbox_rect, dropdown_rect, start_button_rect


async def main():
    global FONT
    # Pygame setup
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Configuration Menu")
    pygame.display.set_icon(pygame.image.load(str(FAVICON_FILE)))
    screen = pygame.display.set_mode((params["WIDTH"], params["HEIGHT"]))
    FONT = pygame.font.Font(None, int(params["WIDTH"] * 0.04))

    # Initialize sliders
    sliders: Dict[str, Tuple[float, pygame.Rect, pygame.Rect]] = {}
    for param, value in params.items():
        if param not in slider_ranges:
            continue

        min_val, max_val = slider_ranges[param]
        sliders[param] = (value, None, None)

    dropdown_selected_index = vehicle_init_options.index(
        params.get("VEHICLE_INIT", "random")
    )

    # Main loop
    running = True
    selected_param = None
    while running:
        await asyncio.sleep(0)
        sliders, checkbox_rect, dropdown_rect, start_button_rect = _draw_menu(
            screen, sliders
        )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Slider selection
                for param, (value, _, handle_rect) in sliders.items():
                    if handle_rect.collidepoint(event.pos):
                        selected_param = param
                        break

                # Checkbox toggle
                if checkbox_rect.collidepoint(event.pos):
                    params["HIGH_DEMAND_ZONE"] = not params["HIGH_DEMAND_ZONE"]

                if dropdown_rect.collidepoint(event.pos):
                    dropdown_selected_index = (dropdown_selected_index + 1) % len(
                        vehicle_init_options
                    )
                    params["VEHICLE_INIT"] = vehicle_init_options[
                        dropdown_selected_index
                    ]

                # Start button
                if start_button_rect.collidepoint(event.pos):
                    # Set high demand zone
                    if params["VEHICLE_INIT"] == "high_demand":
                        params["HIGH_DEMAND_ZONE"] = True

                    # Get slider values
                    for param, (value, _, _) in sliders.items():
                        params[param] = int(value) if param in integer_params else value

                    # Start simulation
                    await run_simulation(
                        width=params["WIDTH"],
                        height=params["HEIGHT"],
                        grid_size=params["GRID_SIZE"],
                        total_vehicles=params["TOTAL_VEHICLES"],
                        battery_depletion_rate=params["BATTERY_DEPLETION_RATE"],
                        recharge_time=params["RECHARGE_TIME"],
                        max_users=params["MAX_USERS"],
                        user_probability=params["USER_PROBABILITY"],
                        fps=params["FPS"],
                        high_demand_zone=params["HIGH_DEMAND_ZONE"],
                        vehicle_init=params["VEHICLE_INIT"],
                    )

                    pygame.display.set_caption("Configuration Menu")
                    screen = pygame.display.set_mode(
                        (params["WIDTH"], params["HEIGHT"])
                    )

            elif event.type == pygame.MOUSEBUTTONUP:
                selected_param = None

            # Slider adjustment
            elif event.type == pygame.MOUSEMOTION and selected_param:
                min_val, max_val = slider_ranges[selected_param]
                x, y, width = (
                    sliders[selected_param][1].x,
                    sliders[selected_param][1].y,
                    sliders[selected_param][1].width,
                )
                rel_x = max(0, min(event.pos[0] - x, width))
                new_value = min_val + (rel_x / width) * (max_val - min_val)

                if selected_param in integer_params:
                    new_value = round(new_value)

                sliders[selected_param] = (
                    new_value,
                    sliders[selected_param][1],
                    sliders[selected_param][2],
                )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    asyncio.run(main())
