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


def _draw_slider(
    screen: pygame.Surface,
    param: str,
    value: float,
    x: int,
    y: int,
    min_val: float,
    max_val: float,
    width: int = 400,
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
        f"{param}: {int(value) if param in integer_params else value:.2f}", True, BLACK
    )
    screen.blit(label, (x, y - 30))

    return slider_rect, handle_rect


def _draw_menu(
    screen: pygame.Surface, sliders: Dict
) -> Tuple[Dict, pygame.Rect, pygame.Rect]:
    screen.fill(WHITE)

    # Title
    title = FONT.render("MAS - Project 3 (Nikethan & Luca)", True, BLACK)
    screen.blit(title, (params["WIDTH"] // 2 - title.get_width() // 2, 20))

    # Sliders
    y_offset = 100
    for i, (param, (value, _, _)) in enumerate(sliders.items()):
        min_val, max_val = slider_ranges[param]
        rect, handle = _draw_slider(
            screen, param, value, 50, y_offset + i * 60, min_val, max_val
        )
        sliders[param] = (value, rect, handle)

    # Checkbox
    checkbox_rect = pygame.Rect(500, y_offset, 20, 20)
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

    checkbox_label = FONT.render("High Demand Zone", True, BLACK)
    screen.blit(checkbox_label, (checkbox_rect.right + 10, checkbox_rect.top))

    # Start button
    start_button = FONT.render("Start", True, WHITE)
    start_button_rect = pygame.Rect(
        params["WIDTH"] // 2 + 130, params["HEIGHT"] - 80, 200, 50
    )
    pygame.draw.rect(screen, BLUE, start_button_rect)
    pygame.draw.rect(screen, BLACK, start_button_rect, 2)
    screen.blit(start_button, (start_button_rect.x + 70, start_button_rect.y + 15))

    return sliders, checkbox_rect, start_button_rect


async def main():
    global FONT
    # Pygame setup
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Configuration Menu")
    pygame.display.set_icon(pygame.image.load(str(FAVICON_FILE)))
    screen = pygame.display.set_mode((params["WIDTH"], params["HEIGHT"]))
    FONT = pygame.font.Font(None, 32)

    # Initialize sliders
    sliders: Dict[str, Tuple[float, pygame.Rect, pygame.Rect]] = {}
    for param, value in params.items():
        if param not in slider_ranges:
            continue

        min_val, max_val = slider_ranges[param]
        sliders[param] = (value, None, None)

    # Main loop
    running = True
    selected_param = None
    while running:
        await asyncio.sleep(0)
        sliders, checkbox_rect, start_button_rect = _draw_menu(screen, sliders)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Slider selection
                for param, (value, slider_rect, handle_rect) in sliders.items():
                    if handle_rect.collidepoint(event.pos):
                        selected_param = param
                        break

                # Checkbox toggle
                if checkbox_rect.collidepoint(event.pos):
                    params["HIGH_DEMAND_ZONE"] = not params["HIGH_DEMAND_ZONE"]

                # Start button
                if start_button_rect.collidepoint(event.pos):
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
