from collections import deque
import heapq


class SearchAgent:
    """
    Search-based agent implementing:
    - Breadth First Search (BFS)
    - Depth First Search (DFS)
    - Uniform Cost Search (UCS)
    """

    # Possible movements
    ACTIONS = [
        ("Up", (0, 1)),
        ("Right", (1, 0)),
        ("Down", (0, -1)),
        ("Left", (-1, 0))
    ]

    def __init__(self):
        # Stores the complete action plan
        self.plan = []

        # Change this to "DFS" or "UCS" to compare algorithms
        self.active_algo = "BFS"

    # ---------------------------------------------------------
    # Get valid neighbouring states
    # ---------------------------------------------------------

    def get_neighbors(self, state, grid_size, walls):
        """
        Return valid neighbouring positions and
        the action required to reach them.
        """

        width, height = grid_size

        x, y = state

        for action, (dx, dy) in self.ACTIONS:

            new_x = x + dx
            new_y = y + dy

            new_state = (new_x, new_y)

            # Check grid boundaries
            if new_x < 0 or new_x >= width:
                continue

            if new_y < 0 or new_y >= height:
                continue

            # Check whether the position is a wall
            if new_state in walls:
                continue

            yield new_state, action

    # ---------------------------------------------------------
    # BFS - Breadth First Search
    # ---------------------------------------------------------

    def bfs_search(self, start, goal, grid_size, walls):
        """
        BFS uses a FIFO queue.

        It explores the shallowest nodes first.
        Since every movement has cost 1, BFS
        finds the shortest path.
        """

        # FIFO queue
        frontier = deque()

        # Store:
        # current state
        # path taken to reach that state
        frontier.append((start, []))

        # Graph Search reached set
        reached = {start}

        while frontier:

            # Remove from the front of the queue
            state, path = frontier.popleft()

            # Goal test
            if state == goal:
                return path

            # Expand current state
            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                # Only visit each state once
                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append(
                        (next_state, new_path)
                    )

        # No path found
        return []

    # ---------------------------------------------------------
    # DFS - Depth First Search
    # ---------------------------------------------------------

    def dfs_search(self, start, goal, grid_size, walls):
        """
        DFS uses a LIFO stack.

        It explores deeper nodes before
        exploring other branches.
        """

        # LIFO stack
        frontier = []

        frontier.append((start, []))

        # Graph Search reached set
        reached = {start}

        while frontier:

            # Remove from the end of the list
            state, path = frontier.pop()

            # Goal test
            if state == goal:
                return path

            # Expand current state
            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                # Avoid revisiting states
                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append(
                        (next_state, new_path)
                    )

        # No path found
        return []

    # ---------------------------------------------------------
    # UCS - Uniform Cost Search
    # ---------------------------------------------------------

    def ucs_search(self, start, goal, grid_size, walls):
        """
        UCS uses a priority queue ordered by
        total path cost g(n).
        """

        # Counter is used to break ties
        # between nodes with the same cost
        counter = 0

        # Priority queue
        #
        # (cost, counter, state, path)
        frontier = [
            (0, counter, start, [])
        ]

        # Store the cheapest known cost
        # for each state
        reached = {
            start: 0
        }

        while frontier:

            # Remove lowest-cost node
            cost, _, state, path = heapq.heappop(
                frontier
            )

            # Goal test
            if state == goal:
                return path

            # Ignore outdated queue entries
            if cost > reached.get(
                state,
                float("inf")
            ):
                continue

            # Expand current state
            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                # Every movement costs 1
                new_cost = cost + 1

                old_cost = reached.get(
                    next_state,
                    float("inf")
                )

                # Only add if this is a cheaper path
                if new_cost < old_cost:

                    reached[next_state] = new_cost

                    counter += 1

                    new_path = path + [action]

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            new_path
                        )
                    )

        # No path found
        return []

    # ---------------------------------------------------------
    # Select the active search algorithm
    # ---------------------------------------------------------

    def search(self, start, goal, percept):
        """
        Run the search algorithm selected
        by self.active_algo.
        """

        grid_size = percept["grid_size"]

        # Convert walls to a set for fast lookup
        walls = set(percept["walls"])

        if self.active_algo == "BFS":

            return self.bfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == "DFS":

            return self.dfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == "UCS":

            return self.ucs_search(
                start,
                goal,
                grid_size,
                walls
            )

        else:

            raise ValueError(
                "Unknown search algorithm: "
                + self.active_algo
            )

    # ---------------------------------------------------------
    # Sense and Act
    # ---------------------------------------------------------

    def sense_and_act(self, percept: dict) -> str:
        """
        If there is no current plan:
        1. Find the closest food.
        2. Run the selected search algorithm.
        3. Store the resulting actions.

        Then execute the first action
        from the plan.
        """

        # Create a new plan only when
        # the current plan is empty
        if not self.plan:

            # Current agent position
            start = tuple(
                percept["agent_pos"]
            )

            # Get all food positions
            all_food = set(
                percept["all_food"]
            )

            # If there is no food left
            if not all_food:
                return "Suck"

            # Find the closest food using
            # Manhattan distance
            goal = min(
                all_food,
                key=lambda food:
                    abs(food[0] - start[0])
                    +
                    abs(food[1] - start[1])
            )

            # Generate complete plan
            self.plan = self.search(
                start,
                goal,
                percept
            )

        # Execute first action
        if self.plan:

            return self.plan.pop(0)

        # Fallback if no path exists
        return "Suck"