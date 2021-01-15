from queue import PriorityQueue
from typing import Tuple

import numpy as np


class TetrisObstacle:
    """ Common interface for Tetris obstacles """
    fill_cell = (4, 3)

    @classmethod
    def gen_filler(cls, x: int, y: int):
        """ Defines all changed points """
        raise NotImplementedError


class Quadrant(TetrisObstacle):
    """
    [
        [..., 100, 100, ...],
        [..., 100, 100, ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x, y), (x + 1, y),
            (x, y + 1), (x + 1, y + 1)
        )


class Line(TetrisObstacle):
    """
    [
        [..., 100, ...],
        [..., 100, ...],
        [..., 100, ...],
        [..., 100, ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x, y),
            (x, y + 1),
            (x, y + 2),
            (x, y + 3)
        )


class Se(TetrisObstacle):
    """
    [
        [..., ..., 100, 100, ...],
        [..., 100, 100, ..., ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x + 1, y), (x + 2, y),
            (x, y + 1), (x + 1, y + 1)
        )


class Zu(TetrisObstacle):
    """
    [
        [..., 100, 100, ..., ...],
        [..., ..., 100, 100, ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x, y), (x + 1, y),
            (x + 1, y + 1), (x + 2, y + 1)
        )


class Lu(TetrisObstacle):
    """
    [
        [..., 100, ..., ...],
        [..., 100, ..., ...],
        [..., 100, ..., ...],
        [..., 100, 100, ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x, y),
            (x, y + 1),
            (x, y + 2),
            (x, y + 3), (x + 1, y + 3),
        )


class Je(TetrisObstacle):
    """
    [
        [..., ..., 100, ...],
        [..., ..., 100, ...],
        [..., ..., 100, ...],
        [..., 100, 100, ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x + 1, y),
            (x + 1, y + 1),
            (x + 1, y + 2),
            (x, y + 3), (x + 1, y + 3),
        )


class Ti(TetrisObstacle):
    """
    [
        [..., 100, 100, 100, ...],
        [..., ..., 100, ..., ...],
    ]
    """

    @classmethod
    def gen_filler(cls, x: int, y: int):
        return (
            (x, y), (x + 1, y), (x + 2, y),
            (x + 1, y + 1),
        )


class Grid:
    """ A flat representation of a grid. Not sure about the grid, but... """

    def __init__(self, step=5):
        self.width = 100
        self.height = 100

        threshold = self.height // step
        self.arrays = np.concatenate([
            np.random.choice(threshold * i, size=(threshold, self.width)) for i in range(1, step + 1)
        ])
        self.arrays = self.arrays.flatten()
        np.random.shuffle(self.arrays)
        self.arrays = np.reshape(self.arrays, (100, 100))

    def add_tetris_obstacles(self, fig_num=5):
        """ Adding randomly chosen Tetris obstacles to the generated and shuffled grid
        """
        calc_fill_indexes = []

        h, w = self.arrays.shape
        step_h, step_w = TetrisObstacle.fill_cell
        cur_h, cur_w = 0, 0
        # Calculate all possible placeholders for the obstacles
        while cur_h < h - 1:
            while cur_w < w - 1:
                calc_fill_indexes.append((cur_h, cur_w))
                cur_w += step_w

            cur_w = 0
            cur_h += step_h

        figures = TetrisObstacle.__subclasses__()
        place_figured = [np.random.choice(figures) for _ in range(fig_num)]
        for figure in place_figured:
            start_point = np.random.choice(len(calc_fill_indexes))

            for coordinate in figure.gen_filler(*calc_fill_indexes[start_point]):
                self.arrays[coordinate] = 100

    def find_easy_path(self, start: Tuple[int, int], finish: Tuple[int, int]):
        variants_queue = PriorityQueue()
        variants_queue.put((self.arrays[start], start, (start,)))
        cost_so_far = {start: self.arrays[start]}

        while not variants_queue.empty():
            cur_prior, current, path = variants_queue.get()

            if current == finish:
                break

            for next_cell in self.find_neighbors(current):
                if self.arrays[next_cell] == 100:
                    continue

                new_cost = cost_so_far[current] + self.arrays[next_cell]

                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + self.get_distance(finish, next_cell)
                    variants_queue.put((priority, next_cell, path + (next_cell,)))

        return path

    @staticmethod
    def get_distance(p_a: Tuple[int, int], p_b: Tuple[int, int]):
        return abs(p_a[0] - p_b[0]) + \
               abs(p_a[1] - p_b[1])

    def find_neighbors(self, point: Tuple[int, int]):
        cur_x, cur_y = point
        return [
            (x, y) for x, y in [
                (cur_x, cur_y + 1), (cur_x - 1, cur_y),
                (cur_x, cur_y - 1), (cur_x + 1, cur_y)
            ] if 0 <= x < self.arrays.shape[0] and 0 <= y < self.arrays.shape[1]
        ]
