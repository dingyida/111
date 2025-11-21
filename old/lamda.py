import re
import pandas as pd
import matplotlib.pyplot as plt

# ——— Configure this! ———
file_path = r"C:\Users\dyd\Desktop\picture\compare_total_lamda.txt"
# ————————————————————

# Prepare containers
data = {
    'Λmax': [],
    'Incremental Update': [],
    'PSO+SLSQP': [],
    'Proportional': [],
    'Reciprocal': []
}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 1) Pull off the leading number as Λmax
        m = re.match(r'^([\d\.]+)', line)
        if not m:
            # if there's a prefix like "Frame 0:", skip or adapt as needed
            continue
        lam = float(m.group(1))

        # 2) Chop off that prefix so we only parse key=val pairs
        rest = line[m.end():]

        # 3) Split on BOTH English and Chinese commas
        parts = re.split(r'[，,]', rest)

        # 4) Build a small dict of the values we care about
        vals = {}
        for part in parts:
            if '=' in part:
                key, val = part.split('=', 1)
                key = key.strip().lower()
                try:
                    vals[key] = float(val)
                except ValueError:
                    pass

        # 5) Append into our lists (use .get(...) in case some line is missing a field)
        data['Λmax'].append(lam)
        data['Incremental Update'].append(vals.get('inc', float('nan')))
        data['PSO+SLSQP'].append(vals.get('full', float('nan')))
        data['Proportional'].append(vals.get('prop', float('nan')))
        data['Reciprocal'].append(vals.get('inv', float('nan')))

# 6) Build a DataFrame and sort by Λmax
df = pd.DataFrame(data)
df = df.sort_values('Λmax')

# Optional check: print how many points we got
print("Loaded points:", len(df))

# ——— Plotting ———
plt.figure(figsize=(10, 6))

plt.plot(df['Λmax'], df['Incremental Update'],
         '-o', label='Incremental Update',
         linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2)

plt.plot(df['Λmax'], df['PSO+SLSQP'],
         '--s', label='PSO+SLSQP',
         linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2)

plt.plot(df['Λmax'], df['Proportional'],
         ':^', label='Proportional',
         linewidth=2, markersize=8)

plt.plot(df['Λmax'], df['Reciprocal'],
         '-.d', label='Reciprocal',
         linewidth=2, markersize=8)

plt.xlabel('Total Upload-Rate Budget (Λmax)', fontsize=14)
plt.ylabel('Global AoI', fontsize=14)

plt.grid(True, linestyle='--', linewidth=0.5)

# Enlarge and stylize the legend
leg = plt.legend(title='Scheme', title_fontsize=13,
                 fontsize=18, loc='upper right', frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.2)

plt.tight_layout()
output_pdf = r"C:\Users\dyd\Desktop\compare2.pdf"
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.show()
