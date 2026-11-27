// Extract move choices at comparable ataris using the same board/search as KataGo.
#include "game/board.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Cells = std::array<std::array<int, 4>, 4>;
using Stones = std::vector<Loc>;

struct Verdict {
  bool valid = false;
  bool full_safe = false;
  bool local_safe = false;
  Loc extension = Board::NULL_LOC;
};

Loc point(int x, int y) { return Location::getLoc(x, y, 19); }

Loc decode(char x, char y) {
  if (x == '-' && y == '-') return Board::PASS_LOC;
  if (x < 'a' || x > 's' || y < 'a' || y > 's') throw std::invalid_argument("bad coordinate");
  return point(x - 'a', y - 'a');
}

Loc one_liberty(const Board& b, Loc stone) {
  Loc current = stone;
  do {
    for (int d = 0; d < 4; ++d) {
      const Loc next = current + b.adj_offsets[d];
      if (b.colors[next] == C_EMPTY) return next;
    }
    current = b.next_in_chain[current];
  } while (current != stone);
  throw std::logic_error("one-liberty chain has no empty neighbor");
}

// Record one point per adjacent friendly chain with two liberties BEFORE the attack.
Stones adjacent_targets(const Board& b, Loc move, Player subject) {
  Stones chains;
  if (move == Board::PASS_LOC) return chains;
  for (int d = 0; d < 4; ++d) {
    const Loc next = move + b.adj_offsets[d];
    if (b.colors[next] != subject || b.getNumLiberties(next) != 2) continue;
    const Loc head = b.chain_head[next];
    if (std::find(chains.begin(), chains.end(), head) == chains.end()) chains.push_back(head);
  }
  return chains;
}

Verdict read_extension(const Board& board, Loc target) {
  if (board.ko_loc != Board::NULL_LOC || board.getNumLiberties(target) != 1) return {};
  const Player subject = board.colors[target];
  const Loc ext = one_liberty(board, target);
  if (!board.isLegal(ext, subject, false) || board.wouldBeCapture(ext, subject)) return {};
  Board full(board);
  full.playMoveAssumeLegal(ext, subject);
  if (full.getNumLiberties(ext) != 2) return {};

  const int x = Location::getX(ext, 19);
  const int y = Location::getY(ext, 19);
  std::vector<Move> placements;
  for (int yy = std::max(0, y - 3); yy <= std::min(18, y + 3); ++yy) {
    for (int xx = std::max(0, x - 3); xx <= std::min(18, x + 3); ++xx) {
      const Loc loc = point(xx, yy);
      if (full.colors[loc] != C_EMPTY) placements.emplace_back(loc, full.colors[loc]);
    }
  }
  Board local(19, 19);
  if (!local.setStonesFailIfNoLibs(placements)) throw std::logic_error("invalid cropped board");
  if (local.getNumLiberties(ext) != 2 || local.getChainSize(ext) != full.getChainSize(ext)) return {};
  std::vector<Loc> scratch, moves;
  const bool full_safe = !full.searchIsLadderCapturedAttackerFirst2Libs(ext, scratch, moves);
  const bool local_safe = !local.searchIsLadderCapturedAttackerFirst2Libs(ext, scratch, moves);
  return {true, full_safe, local_safe, ext};
}

struct Stats {
  Cells cells{};
  int own_turns = 0;
  int single_threats = 0;
  int continuation = 0;
  bool valid = true;
};

Stats extract(Player subject, const std::string& setup, const std::string& sequence) {
  Board b(19,19);
  if (setup != "-") {
    if (setup.size() % 2 != 0) throw std::invalid_argument("incomplete handicap setup");
    for (size_t i = 0; i < setup.size(); i += 2)
      if (!b.setStone(decode(setup[i], setup[i + 1]), P_BLACK)) throw std::invalid_argument("illegal setup");
  }
  Stats stats;
  Stones targets;
  Loc chased = Board::NULL_LOC;
  Player expected = setup == "-" ? P_BLACK : P_WHITE;
  if (sequence.empty() || sequence.size() % 3 != 0) throw std::invalid_argument("incomplete game");
  for (size_t i = 0; i < sequence.size(); i += 3) {
    const Player pla = sequence[i] == 'B' ? P_BLACK : P_WHITE;
    const Loc move = decode(sequence[i + 1], sequence[i + 2]);
    if ((sequence[i] != 'B' && sequence[i] != 'W') || pla != expected || !b.isLegal(move, pla, false)) {
      // Exclude the whole game, including any events seen before the first illegal move.
      Stats bad;
      bad.valid = false;
      return bad;
    }
    expected = getOpp(pla);
    if (pla != subject) {
      targets = adjacent_targets(b, move, subject);
      b.playMoveAssumeLegal(move, pla);
      targets.erase(std::remove_if(targets.begin(), targets.end(), [&](Loc stone) {
        return b.colors[stone] != subject || b.getNumLiberties(stone) != 1;
      }), targets.end());
      continue;
    }
    ++stats.own_turns;
    Loc next_chased = Board::NULL_LOC;
    if (targets.size() == 1) {
      const Loc target = targets.front();
      ++stats.single_threats;
      const bool same_chain = chased != Board::NULL_LOC && b.colors[chased] == subject
                           && b.chain_head[chased] == b.chain_head[target];
      if (same_chain) {
        ++stats.continuation;
        if (move == one_liberty(b, target)) next_chased = move;
      } else {
        const Verdict v = read_extension(b, target);
        if (v.valid) {
          const bool played = move == v.extension;
          const int cell = 2 * int(v.local_safe) + int(b.getChainSize(target) > 1);
          auto& counts = stats.cells[cell];
          ++counts[0];
          counts[1] += int(v.full_safe);
          counts[2] += int(played);
          counts[3] += int(played && v.full_safe);
          if (played) next_chased = move;
        }
      }
    }
    chased = next_chased;
    b.playMoveAssumeLegal(move, pla);
    targets.clear();
  }
  return stats;
}

void self_check() {
  Board b(19, 19);
  const Loc target = point(4,4);
  b.setStone(target, P_BLACK);
  for (auto loc : {point(3,4), point(4,3), point(5,4), point(3,5)}) b.setStone(loc, P_WHITE);
  const auto before = b.pos_hash;
  const Verdict start = read_extension(b, target);
  if (!start.valid || start.full_safe || start.local_safe || start.extension != point(4,5) || before != b.pos_hash)
    throw std::logic_error("standard chased ladder or counterfactual board restoration failed");
  // A distant defender on the path must flip the full-board reading only.
  bool flipped = false;
  for (int y = 9; y <= 13 && !flipped; ++y) {
    for (int x = 8; x <= 12 && !flipped; ++x) {
      Board with_breaker(b);
      with_breaker.setStone(point(x,y), P_BLACK);
      const Verdict v = read_extension(with_breaker, target);
      flipped = v.valid && v.full_safe && !v.local_safe;
    }
  }
  if (!flipped) throw std::logic_error("distant ladder breaker did not change the expected field");
  // Interleave unrelated legal filler moves: each game has the same atari but a different actual reply.
  const std::string prefix = "BeeWdeBssWedBsrWdfBsqWfe";
  const Stats extend = extract(P_BLACK, "-", prefix + "Bef");
  const Stats ignore = extract(P_BLACK, "-", prefix + "Bqq");
  const Stats illegal = extract(P_BLACK, "-", prefix + "Bee");
  if (!extend.valid || !ignore.valid || illegal.valid || extend.cells[0] != std::array<int,4>{1,0,1,0}
      || ignore.cells[0] != std::array<int,4>{1,0,0,0})
    throw std::logic_error("turn ordering, observed choice, or exclusion failed");
  std::cout << "Board, ladder breaker, observed choice, and invalid-game checks passed\n";
}
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  Board::initHash();
  if (argc == 2 && std::string(argv[1]) == "--self-check") { self_check(); return 0; }
  int id;
  char color;
  std::string setup, sequence;
  while (std::cin >> id >> color >> setup >> sequence) {
    if (color != 'B' && color != 'W') throw std::invalid_argument("invalid subject");
    const Stats s = extract(color == 'B' ? P_BLACK : P_WHITE, setup, sequence);
    std::cout << id << ' ' << int(s.valid) << ' ' << s.own_turns << ' ' << s.single_threats << ' ' << s.continuation;
    for (const auto& cell : s.cells) for (int count : cell) std::cout << ' ' << count;
    std::cout << '\n';
  }
  if (!std::cin.eof()) return 2;
}
