// Ask which of two legal first ataris catches the enemy chain just played.
#include "game/board.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Points = std::vector<Loc>;
using Cells = std::array<std::array<int, 6>, 6>;

Loc point(int x, int y) { return Location::getLoc(x, y, 19); }

Loc decode(char a, char b) {
  if (a == '-' && b == '-') return Board::PASS_LOC;
  if (a < 'a' || a > 's' || b < 'a' || b > 's') throw std::invalid_argument("coordinate");
  return point(a-'a', b-'a');
}

Points liberties(const Board& board, Loc target) {
  Points result;
  Loc stone = target;
  do {
    for (int d = 0; d < 4; ++d) {
      const Loc n = stone + board.adj_offsets[d];
      if (board.colors[n] == C_EMPTY && std::find(result.begin(), result.end(), n) == result.end())
        result.push_back(n);
    }
    stone = board.next_in_chain[stone];
  } while (stone != target);
  std::sort(result.begin(), result.end());
  return result;
}

int edge(Loc loc) {
  const int x = Location::getX(loc, 19), y = Location::getY(loc, 19);
  return std::min({x, y, 18-x, 18-y});
}

struct Choice {
  bool valid = false;
  int cell = 0;
  Loc ref = Board::NULL_LOC, winner = Board::NULL_LOC;
  Points wings;
};

Choice assess(Board& full, Loc target) {
  if (full.ko_loc != Board::NULL_LOC || full.getNumLiberties(target) != 2) return {};
  const Player attacker = getOpp(full.colors[target]);
  const Points wings = liberties(full, target);
  if (wings.size() != 2) throw std::logic_error("chain liberties");
  for (Loc l : wings) if (!full.isLegal(l,attacker,false) || full.wouldBeCapture(l,attacker)) return {};
  Points buf, moves;
  if (!full.searchIsLadderCapturedAttackerFirst2Libs(target, buf, moves) || moves.size() != 1) return {};
  const Loc winner = moves.front();
  const int x = Location::getX(target,19), y = Location::getY(target,19);
  std::vector<Move> placements;
  for (int yy=std::max(0,y-3); yy<=std::min(18,y+3); ++yy)
    for (int xx=std::max(0,x-3); xx<=std::min(18,x+3); ++xx) {
      const Loc l=point(xx,yy);
      if (full.colors[l] != C_EMPTY) placements.emplace_back(l,full.colors[l]);
    }
  Board local(19,19);
  if (!local.setStonesFailIfNoLibs(placements)) throw std::logic_error("crop");
  if (local.getChainSize(target) != full.getChainSize(target) || local.getNumLiberties(target) != 2 ||
      liberties(local,target) != wings) return {};
  for (Loc l : wings) if (!local.isLegal(l,attacker,false) || local.wouldBeCapture(l,attacker)) return {};
  moves.clear();  // KataGo leaves the output list untouched when there is no ladder.
  local.searchIsLadderCapturedAttackerFirst2Libs(target, buf, moves);
  const Loc tie_ref = edge(wings[0]) <= edge(wings[1]) ? wings[0] : wings[1];
  const Loc ref = moves.size() == 1 ? moves.front() : tie_ref;
  return {true, 2*int(moves.size()) + int(full.getChainSize(target) > 1), ref, winner, wings};
}

struct Counts {
  bool valid = true;
  int turns = 0;
  Cells cells{};
};

Counts extract(Player subject, const std::string& setup, const std::string& seq) {
  Board board(19,19);
  if (setup != "-") {
    if (setup.size()%2 != 0) throw std::invalid_argument("handicap coordinates");
    for (size_t i=0; i<setup.size(); i+=2)
      if (!board.setStone(decode(setup[i], setup[i+1]), P_BLACK)) throw std::invalid_argument("setup");
  }
  Player expected = setup == "-" ? P_BLACK : P_WHITE;
  Loc last_enemy = Board::NULL_LOC, chased = Board::NULL_LOC;
  Counts result;
  if (seq.empty() || seq.size()%3 != 0) throw std::invalid_argument("missing moves");
  for (size_t i=0; i<seq.size(); i+=3) {
    const Player pla = seq[i]=='B' ? P_BLACK : P_WHITE;
    const Loc move = decode(seq[i+1], seq[i+2]);
    if ((seq[i]!='B' && seq[i]!='W') || pla!=expected || !board.isLegal(move,pla,false)) {
      Counts invalid;
      invalid.valid=false;
      return invalid;
    }
    expected=getOpp(pla);
    if (pla==subject) {
      ++result.turns;
      Loc newly_chased=Board::NULL_LOC;
      if (last_enemy!=Board::NULL_LOC && last_enemy!=Board::PASS_LOC && board.getNumLiberties(last_enemy)==2) {
        const bool repeat = chased!=Board::NULL_LOC && board.colors[chased]==getOpp(subject) &&
                            board.chain_head[chased]==board.chain_head[last_enemy];
        if (repeat) {
          auto wings=liberties(board,last_enemy);
          if (std::find(wings.begin(),wings.end(),move)!=wings.end()) newly_chased=last_enemy;
        } else {
          const Choice c=assess(board,last_enemy);
          if (c.valid) {
            const bool attempted=std::find(c.wings.begin(),c.wings.end(),move)!=c.wings.end();
            const bool good=c.winner==c.ref, preferred=move==c.ref;
            auto& bins=result.cells[c.cell];
            ++bins[0]; bins[1]+=good; bins[2]+=attempted; bins[3]+=attempted && good;
            bins[4]+=preferred; bins[5]+=preferred && good;
            if (attempted) newly_chased=last_enemy;
          }
        }
      }
      chased=newly_chased;
    } else {
      last_enemy=move;
    }
    board.playMoveAssumeLegal(move,pla);
  }
  return result;
}

void check() {
  Board b(19,19);
  const std::string prefix="BeeWdeBssWedBsrWdfBsqWfeBef";
  for (size_t i=0; i<prefix.size(); i+=3)
    if (!b.playMove(decode(prefix[i+1],prefix[i+2]),prefix[i]=='B'?P_BLACK:P_WHITE,false))
      throw std::logic_error("fixture illegal");
  const Loc target=point(4,5);
  const auto hash=b.pos_hash;
  const Choice c=assess(b,target);
  if (!c.valid || c.wings.size()!=2 || b.pos_hash!=hash) throw std::logic_error("fixture lacks a unique attack");
  const Counts base=extract(P_WHITE,"-",prefix);
  for (auto wing:c.wings) {
    Board trial(b);
    trial.playMoveAssumeLegal(wing,P_WHITE);
    Points scratch;
    const bool works=trial.searchIsLadderCaptured(target,true,scratch);
    if (works != (wing==c.winner)) throw std::logic_error("first move does not match separate continuation search");
    const std::string suffix={'W',char('a'+Location::getX(wing,19)),char('a'+Location::getY(wing,19))};
    const Counts found=extract(P_WHITE,"-",prefix+suffix);
    auto d=found.cells[c.cell];
    for (size_t i=0;i<d.size();++i) d[i]-=base.cells[c.cell][i];
    if (!found.valid || d[0]!=1 || d[2]!=1 || d[4]!=(wing==c.ref) || d[2]-d[3]-d[4]+2*d[5]!=int(works))
      throw std::logic_error("replay direction is wrong");
  }
  const Counts pass=extract(P_WHITE,"-",prefix+"W--"), invalid=extract(P_WHITE,"-",prefix+"Wef");
  if (!pass.valid || invalid.valid || pass.cells[c.cell][2]!=base.cells[c.cell][2])
    throw std::logic_error("pass or illegal move");
  std::cout << "Unique ladder attack, independent direction search, both choices, pass and invalid-game checks passed\n";
}
}

int main(int argc,char** argv) {
  Board::initHash();
  if (argc==2 && std::string(argv[1])=="--self-check") { check(); return 0; }
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int id;
  char color;
  std::string setup,sequence;
  while (std::cin>>id>>color>>setup>>sequence) {
    if (color!='B' && color!='W') throw std::invalid_argument("subject");
    const Counts c=extract(color=='B'?P_BLACK:P_WHITE,setup,sequence);
    std::cout << id << ' ' << int(c.valid) << ' ' << c.turns;
    for (const auto& cell:c.cells) for (int count:cell) std::cout << ' ' << count;
    std::cout << '\n';
  }
  return std::cin.eof()?0:2;
}
