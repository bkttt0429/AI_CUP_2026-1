#include "game/board.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int side = 19;
Loc xy(int x,int y) { return Location::getLoc(x,y,side); }
Loc decode(char x,char y) {
  if (x=='-' && y=='-') return Board::PASS_LOC;
  if (x<'a'||x>'s'||y<'a'||y>'s') throw std::invalid_argument("coordinate");
  return xy(x-'a',y-'a');
}
int packed(Loc p) {
  return p==Board::PASS_LOC ? 361 : Location::getY(p,side)*19+Location::getX(p,side);
}
int distance(Loc a,Loc b) {
  return std::max(std::abs(Location::getX(a,side)-Location::getX(b,side)),
                  std::abs(Location::getY(a,side)-Location::getY(b,side)));
}
std::vector<Loc> touched(const Board& board,Loc move,Player victim) {
  std::vector<Loc> heads;
  if (move!=Board::PASS_LOC) for (int k=0;k<4;++k) {
    Loc neighbor=move+board.adj_offsets[k];
    if (board.colors[neighbor]!=victim || board.getNumLiberties(neighbor)!=2) continue;
    Loc h=board.chain_head[neighbor];
    if (std::find(heads.begin(),heads.end(),h)==heads.end()) heads.push_back(h);
  }
  return heads;
}
Loc last_lib(const Board& b,Loc chain) {
  auto stone=chain;
  do {
    for (int k=0;k<4;++k) if (b.colors[stone+b.adj_offsets[k]]==C_EMPTY) return stone+b.adj_offsets[k];
    stone=b.next_in_chain[stone];
  } while(stone!=chain);
  throw std::logic_error("missing liberty");
}

struct Escape { bool valid=false, full=false, local=false; Loc move=Board::NULL_LOC; };
Escape escape(const Board& b,Loc target) {
  if (b.ko_loc!=Board::NULL_LOC) return {};
  const Loc loc=last_lib(b,target);
  Player mover=b.colors[target];
  if (!b.isLegal(loc,mover,false) || b.wouldBeCapture(loc,mover)) return {};
  Board full=b;
  full.playMoveAssumeLegal(loc,mover);
  if (full.getNumLiberties(loc)!=2) return {};
  int x=Location::getX(loc,side), y=Location::getY(loc,side);
  std::vector<Move> stones;
  for (int j=std::max(y-3,0);j<=std::min(y+3,18);++j)
    for (int i=std::max(x-3,0);i<=std::min(x+3,18);++i) {
      Loc l=xy(i,j);
      if (full.colors[l]!=C_EMPTY) stones.emplace_back(l,full.colors[l]);
    }
  Board local(side,side);
  if (!local.setStonesFailIfNoLibs(stones)) throw std::logic_error("invalid crop");
  if (local.getNumLiberties(loc)!=2 || local.getChainSize(loc)!=full.getChainSize(loc)) return {};
  std::vector<Loc> buf,winning;
  bool good=!full.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,winning);
  bool local_good=!local.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,winning);
  return {true,good,local_good,loc};
}
int captures(const Board& b,Player mover) { return mover==P_BLACK? b.numWhiteCaptures : b.numBlackCaptures; }
struct Response {
  int kind=0, immediate_taken=0, new_atari=0, near=0, immediate_lib=0, replies=0;
  int lost_first=-1,lost_second=-1,taken_window=-1,final_lib=-1;
};

Response follow(Board b,Loc target,Loc ext,const std::vector<Move>& moves,int t) {
  Response o;
  Player mover=b.colors[target],enemy=getOpp(mover);
  auto move=moves[t].loc;
  auto threatened=touched(b,move,enemy);
  const int taken0=captures(b,mover);
  b.playMoveAssumeLegal(move,mover);
  o.immediate_taken=captures(b,mover)-taken0;
  o.immediate_lib=b.getNumLiberties(target);
  for (Loc h:threatened) if (b.colors[h]==enemy && b.getNumLiberties(h)==1) ++o.new_atari;
  o.near=move!=Board::PASS_LOC && distance(ext,move)<=3;
  if (o.immediate_lib>=2) o.kind=0;
  else if (o.immediate_taken>0) o.kind=1;
  else if (o.new_atari>0) o.kind=2;
  else if (move==Board::PASS_LOC) o.kind=5;
  else o.kind=o.near?3:4;
  bool lost=false;
  for (int h=1;h<=3 && t+h<int(moves.size());++h) {
    const auto& m=moves[t+h];
    if (!b.playMove(m.loc,m.pla,false)) throw std::logic_error("illegal continuation");
    lost = lost || (b.colors[target]!=mover);
    if (h==1) { o.replies=1; o.lost_first=lost; }
    if (h==3) {
      o.replies=2; o.lost_second=lost; o.taken_window=captures(b,mover)-taken0;
      o.final_lib=lost?0:b.getNumLiberties(target);
    }
  }
  return o;
}
void test(bool x,const char* message) { if(!x) throw std::logic_error(message); }
void self_test() {
  Board b(side,side);
  Loc target=xy(4,4),ext=xy(4,5);
  b.setStone(target,P_BLACK);
  for (Loc l:{xy(3,4),xy(4,3),xy(5,4)}) b.setStone(l,P_WHITE);
  auto r=follow(b,target,ext,{{xy(16,16),P_BLACK},{ext,P_WHITE},{xy(16,17),P_BLACK},{xy(15,17),P_WHITE}},0);
  test(r.kind==4 && r.lost_first==1 && r.lost_second==1 && r.final_lib==0 && r.taken_window==0,"captured original");
  r=follow(b,target,ext,{{xy(4,7),P_BLACK}},0);
  test(r.kind==3 && r.lost_second==-1 && r.replies==0,"near alternative or censored");
  r=follow(b,target,ext,{{Board::PASS_LOC,P_BLACK},{xy(16,16),P_WHITE}},0);
  test(r.kind==5 && r.lost_first==0 && r.replies==1 && r.taken_window==-1,"pass and one reply");
  Board rescue=b;
  for (Loc l:{xy(5,3),xy(5,5)}) rescue.setStone(l,P_BLACK);
  r=follow(rescue,target,ext,{{xy(6,4),P_BLACK}},0);
  test(r.kind==0 && r.immediate_taken==1 && r.immediate_lib==2,"rescue by a different capture");
  Board other=b;
  other.setStone(xy(12,12),P_WHITE);
  for (Loc l:{xy(11,12),xy(12,11)}) other.setStone(l,P_BLACK);
  r=follow(other,target,ext,{{xy(13,12),P_BLACK}},0);
  test(r.kind==2 && r.new_atari==1 && !r.near,"remote newly-created threat");
  other.setStone(xy(13,12),P_BLACK);
  r=follow(other,target,ext,{{xy(12,13),P_BLACK}},0);
  test(r.kind==1 && r.immediate_taken==1 && r.immediate_lib==1,"remote capture");
  test(b.getNumLiberties(target)==1,"branch restored");
}
void extract(int id,Player subject,const std::string& setup,const std::string& seq,std::ostream& stats) {
  Board b(side,side);
  if (setup!="-") for(size_t i=0;i<setup.size();i+=2)
    test(b.setStone(decode(setup.at(i),setup.at(i+1)),P_BLACK),"bad setup");
  std::vector<Move> moves;
  Player expected=setup=="-"?P_BLACK:P_WHITE;
  for(size_t i=0;i<seq.size();i+=3) {
    auto p=seq[i]=='B'?P_BLACK:P_WHITE;
    test((seq[i]=='B'||seq[i]=='W') && p==expected,"bad turn");
    moves.emplace_back(decode(seq.at(i+1),seq.at(i+2)),p);
    expected=getOpp(expected);
  }
  Loc chase=Board::NULL_LOC;
  std::vector<Loc> groups;
  int own_turns=0;
  for(int t=0;t<int(moves.size());++t) {
    const Loc move=moves[t].loc;
    const Player p=moves[t].pla;
    test(b.isLegal(move,p,false),"illegal game");
    if(p!=subject) {
      groups=touched(b,move,subject);
      b.playMoveAssumeLegal(move,p);
      groups.erase(std::remove_if(groups.begin(),groups.end(),[&](Loc h) {
        return b.colors[h]!=subject || b.getNumLiberties(h)!=1;
      }),groups.end());
      continue;
    }
    ++own_turns;
    auto next=Board::NULL_LOC;
    if(groups.size()==1) {
      Loc h=groups[0];
      const bool repeated=chase!=Board::NULL_LOC && b.colors[chase]==subject && b.chain_head[chase]==b.chain_head[h];
      if(repeated) {
        if(move==last_lib(b,h)) next=move;
      } else {
        auto e=escape(b,h);
        if(e.valid && move==e.move) next=move;
        if(e.valid && e.full && move!=e.move) {
          auto r=follow(b,h,e.move,moves,t);
          std::array<int,18> row={id,t,int(subject==P_WHITE),packed(h),packed(e.move),packed(move),
             b.getChainSize(h),int(e.local),r.kind,r.immediate_taken,r.new_atari,r.near,r.immediate_lib,
             r.replies,r.lost_first,r.lost_second,r.taken_window,r.final_lib};
          for(int value:row) std::cout<<value<<' ';
          std::cout<<'\n';
        }
      }
    }
    chase=next;
    b.playMoveAssumeLegal(move,p);
    groups.clear();
  }
  stats<<id<<' '<<own_turns<<'\n';
}
}
int main(int argc,char** argv) {
  Board::initHash();
  if(argc==2 && std::string(argv[1])=="--self-test") { self_test(); std::cout<<"All six alternative classes, capture, horizon and board restoration passed\n"; return 0; }
  if(argc!=2) return 2;
  std::ofstream stats(argv[1]); if(!stats) return 2;
  std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
  int id; char color; std::string setup,seq;
  while(std::cin>>id>>color>>setup>>seq) {
    test(color=='B'||color=='W',"colour");
    extract(id,color=='B'?P_BLACK:P_WHITE,setup,seq,stats);
  }
  return std::cin.eof() && stats.good()?0:2;
}
