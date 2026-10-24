import sys, csv, random
from sgfmill import sgf, boards
csv.field_size_limit(1 << 30)
random.seed(0)
N = int(sys.argv[1]); tot = bad = kov = 0; errs = {}
for path in sys.argv[2:]:
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for row in random.sample(rows, N):
        tot += 1
        try:
            g = sgf.Sgf_game.from_string(row['sgf_content'])
            b = boards.Board(19); ko = None
            for node in g.get_main_sequence()[1:]:
                c, m = node.get_move()
                if m is None: raise ValueError('pass')
                if m == ko: kov += 1
                ko = b.play(m[0], m[1], c)
        except Exception as e:
            bad += 1; k = type(e).__name__ + ':' + str(e)[:50]; errs[k] = errs.get(k, 0) + 1
    print(path.split('/')[-1], 'sampled', N, 'bad so far', bad, flush=True)
print('TOTAL', tot, 'parse/illegal', bad, 'simple-ko violations', kov, errs)
