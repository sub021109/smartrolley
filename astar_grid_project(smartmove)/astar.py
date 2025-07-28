import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0 + heuristic(start, goal), 0, start, [])]
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current in visited: continue
        visited.add(current)
        path = path + [current]

        if current == goal:
            return path

        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    heapq.heappush(open_set, (g+1+heuristic((nx,ny), goal), g+1, (nx,ny), path))
    return []
