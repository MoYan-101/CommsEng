import os
import pandas as pd
from collections import defaultdict, deque

# === Load data ===
input_path = os.path.join(os.path.dirname(__file__), "Main_20260128.csv")
df = pd.read_csv(input_path)
df.columns = [c.strip() for c in df.columns]

col1 = "Promoter 1"
col2 = "Promoter 2"
ratio1 = "Promoter 1 ratio (Promoter 1:Cu)"
ratio2 = "Promoter 2 ratio (Promoter 2:Cu)"
df["row_id"] = df.index  # Track removed rows

# === Build graph (one row = one edge) ===
edges = []
for idx, row in df.iterrows():
    p1, p2 = row[col1], row[col2]
    if p1 != "Null" and p2 != "Null" and p1 != p2:
        edges.append((p1, p2, idx))

# === Try coloring and collect conflict edges ===
def find_conflicts(edges):
    graph = defaultdict(list)
    for u, v, _ in edges:
        graph[u].append(v)
        graph[v].append(u)

    color = {}
    conflicts = set()
    for node in graph:
        if node in color:
            continue
        queue = deque([node])
        color[node] = 0
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    for e in edges:
                        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                            conflicts.add(e[2])
    return conflicts, color

# === Greedy conflict-edge removal ===
conflict_indices = set()
for _ in range(100):  # Up to 100 iterations
    conflicts, color_map = find_conflicts(edges)
    if not conflicts:
        break
    conflict_indices.update(conflicts)
    edges = [e for e in edges if e[2] not in conflicts]

# === Split conflict and clean rows ===
df_conflict = df[df["row_id"].isin(conflict_indices)].copy()  # Keep conflict rows
df_clean    = df[~df["row_id"].isin(conflict_indices)].copy()
df_clean.drop(columns="row_id", inplace=True)
df_conflict.drop(columns="row_id", inplace=True)              # Drop helper column

base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.dirname(input_path)
conflict_path = os.path.join(output_dir, f"{base_name}_conflict_rows.csv")
clean_path = os.path.join(output_dir, f"{base_name}_cleansed.csv")

# Save conflict rows separately
df_conflict.to_csv(conflict_path, index=False)
print(f"Saved {len(df_conflict)} conflict rows to '{conflict_path}'")
print(f"\n Removed conflict rows: {len(conflict_indices)}")

# === Recolor clean graph to reorder columns ===
final_graph = defaultdict(list)
for idx, row in df_clean.iterrows():
    p1, p2 = row[col1], row[col2]
    if p1 != "Null":
        final_graph[p1]
    if p2 != "Null":
        final_graph[p2]
    if p1 != "Null" and p2 != "Null":
        final_graph[p1].append(p2)
        final_graph[p2].append(p1)

color = {}
for prom in final_graph:
    if prom in color:
        continue
    color[prom] = 0
    queue = deque([prom])
    while queue:
        u = queue.popleft()
        for v in final_graph[u]:
            if v not in color:
                color[v] = 1 - color[u]
                queue.append(v)

# === Reorder promoter columns and ratios ===
for idx, row in df_clean.iterrows():
    p1, p2 = row[col1], row[col2]
    r1, r2 = row[ratio1], row[ratio2]

    if p1 != "Null" and p2 != "Null":
        if color[p1] == 1:
            df_clean.at[idx, col1] = p2
            df_clean.at[idx, col2] = p1
            df_clean.at[idx, ratio1] = r2
            df_clean.at[idx, ratio2] = r1
    elif p1 == "Null" and p2 != "Null":
        if color.get(p2, 0) == 0:
            df_clean.at[idx, col1] = p2
            df_clean.at[idx, col2] = "Null"
            df_clean.at[idx, ratio1] = r2
            df_clean.at[idx, ratio2] = r1
    elif p2 == "Null" and p1 != "Null":
        if color.get(p1, 0) == 1:
            df_clean.at[idx, col1] = "Null"
            df_clean.at[idx, col2] = p1
            df_clean.at[idx, ratio1] = r2
            df_clean.at[idx, ratio2] = r1

# === Check full separation ===
p1_set = set(df_clean[col1]) - {"Null"}
p2_set = set(df_clean[col2]) - {"Null"}
conflict = p1_set & p2_set

if conflict:
    print(f"Remaining conflicting elements: {conflict}")
else:
    print("Promoter 1 and Promoter 2 are fully separated")

# === Print unique values per column ===
p1_all = sorted(set(df_clean[col1]))
p2_all = sorted(set(df_clean[col2]))

print("\nUnique values in Promoter 1:")
print(p1_all)
print("\nUnique values in Promoter 2:")
print(p2_all)

# === Save final output ===
df_clean.to_csv(clean_path, index=False)
print(f"\n File saved as '{clean_path}'")
