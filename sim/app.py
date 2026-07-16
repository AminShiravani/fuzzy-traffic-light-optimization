"""
app.py

Main entry point of the traffic simulation viewer.
"""

import pygame

from loader import load_simulation
from simulation import SimulationPlayer
from renderer import Renderer, WIDTH, HEIGHT
from pathlib import Path

FPS = 30


def main():

    # ------------------------------
    # Initialize pygame
    # ------------------------------

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Traffic Light Simulation")

    clock = pygame.time.Clock()

    # ------------------------------
    # Load simulation
    # ------------------------------


    BASE_DIR = Path(__file__).parent

    simulation = load_simulation(
        BASE_DIR / "data" / "baseline.json"
    )
    player = SimulationPlayer(simulation)

    renderer = Renderer(screen)

    paused = False

    running = True

    # ------------------------------
    # Main Loop
    # ------------------------------

    while running:

        #
        # Events
        #

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                #
                # Quit
                #

                if event.key == pygame.K_ESCAPE:
                    running = False

                #
                # Pause
                #

                elif event.key == pygame.K_SPACE:

                    paused = not paused

                #
                # Restart
                #

                elif event.key == pygame.K_r:

                    player.restart()

                #
                # Next frame manually
                #

                elif event.key == pygame.K_RIGHT:

                    player.next_frame()

                #
                # Previous frame manually
                #

                elif event.key == pygame.K_LEFT:

                    player.previous_frame()

        #
        # Automatic playback
        #

        if not paused:

            if not player.is_finished():

                player.next_frame()

        #
        # Draw
        #

        state = player.get_state()

        renderer.draw(state)

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()