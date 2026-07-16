"""
renderer.py

Draws the traffic simulation.

Everything here is visual only.
No fuzzy logic.
No optimization.
"""

import pygame


WIDTH = 1000
HEIGHT = 700

ROAD_WIDTH = 120

CAR_RADIUS = 8
CAR_SPACING = 22

BACKGROUND = (235, 235, 235)

ROAD = (70, 70, 70)

RED = (220, 50, 50)
GREEN = (50, 200, 50)

BLUE = (40, 120, 240)

BLACK = (20, 20, 20)

WHITE = (255, 255, 255)


class Renderer:

    def __init__(self, screen):

        self.screen = screen

        self.font = pygame.font.SysFont(None, 28)

        self.big_font = pygame.font.SysFont(None, 36)

        self.center_x = WIDTH // 2
        self.center_y = HEIGHT // 2

    # ----------------------------------------------------

    def draw(self, state):

        self.screen.fill(BACKGROUND)

        self.draw_roads()

        self.draw_traffic_lights(state)

        self.draw_cars(state)

        self.draw_information(state)

    # ----------------------------------------------------

    def draw_roads(self):

        pygame.draw.rect(
            self.screen,
            ROAD,
            (
                self.center_x - ROAD_WIDTH // 2,
                0,
                ROAD_WIDTH,
                HEIGHT,
            ),
        )

        pygame.draw.rect(
            self.screen,
            ROAD,
            (
                0,
                self.center_y - ROAD_WIDTH // 2,
                WIDTH,
                ROAD_WIDTH,
            ),
        )

    # ----------------------------------------------------

    def draw_traffic_lights(self, state):

        if state.light1 == 1:
            color1 = GREEN
        else:
            color1 = RED

        if state.light2 == 1:
            color2 = GREEN
        else:
            color2 = RED

        pygame.draw.circle(
            self.screen,
            color1,
            (self.center_x - 40, self.center_y - 90),
            12,
        )

        pygame.draw.circle(
            self.screen,
            color2,
            (self.center_x + 90, self.center_y + 40),
            12,
        )

    # ----------------------------------------------------

    def draw_cars(self, state):

        #
        # Road 1 (vertical)
        #

        start_y = self.center_y - 130

        for i in range(int(state.queue1)):

            pygame.draw.circle(
                self.screen,
                BLUE,
                (
                    self.center_x - 20,
                    start_y - i * CAR_SPACING,
                ),
                CAR_RADIUS,
            )

        #
        # Road 2 (horizontal)
        #

        start_x = self.center_x + 130

        for i in range(int(state.queue2)):

            pygame.draw.circle(
                self.screen,
                BLUE,
                (
                    start_x + i * CAR_SPACING,
                    self.center_y + 20,
                ),
                CAR_RADIUS,
            )

    # ----------------------------------------------------

    def draw_information(self, state):

        lines = [

            f"Time : {state.time}",

            f"Frame : {state.frame}",

            f"Queue Road 1 : {state.queue1}",

            f"Queue Road 2 : {state.queue2}",

            f"Green Time Road 1 : {state.green1:.1f}",

            f"Green Time Road 2 : {state.green2:.1f}",

        ]

        y = 20

        for text in lines:

            img = self.font.render(text, True, BLACK)

            self.screen.blit(img, (20, y))

            y += 28

        title = self.big_font.render(
            "Traffic Light Simulation",
            True,
            BLACK,
        )

        self.screen.blit(title, (320, 15))