// The first replay is a frozen experiment. Import its board and eligibility routines verbatim.
#define main archived_alternative_cli
#include "read_alternatives.cpp"
#undef main
#include <map>

namespace {
std::map<Loc,int> other_rescues(const Board& b,Loc target,Loc direct) {
  const Player own=b.colors[target],enemy=getOpp(own);
  std::vector<Loc> candidates;
  Loc stone=target;
  do {
    for(int k=0;k<4;++k) {
      Loc other=stone+b.adj_offsets[k];
      if(b.colors[other]!=enemy || b.getNumLiberties(other)!=1) continue;
      Loc last=last_lib(b,other);
      if(last!=direct && std::find(candidates.begin(),candidates.end(),last)==candidates.end()) candidates.push_back(last);
    }
    stone=b.next_in_chain[stone];
  } while(stone!=target);
  std::map<Loc,int> offers;
  for(Loc candidate:candidates) {
    Board copy=b;
    if(!copy.playMove(candidate,own,false)) continue;
    if(copy.getNumLiberties(target)>=2) {
      int taken=captures(copy,own)-captures(b,own);
      test(taken>0,"a non-connecting rescue should capture");
      offers[candidate]=taken;
    }
  }
  return offers;
}
std::map<Loc,int> exhaustive(const Board& b,Loc target,Loc direct) {
  std::map<Loc,int> all;
  Player own=b.colors[target];
  for(int y=0;y<side;++y) for(int x=0;x<side;++x) {
    Loc move=xy(x,y);
    if(move==direct || !b.isLegal(move,own,false)) continue;
    Board copy=b; copy.playMoveAssumeLegal(move,own);
    if(copy.getNumLiberties(target)>=2) all[move]=captures(copy,own)-captures(b,own);
  }
  return all;
}
void read_menu(int game,Player who,const std::string& setup,const std::string& seq,std::ostream& stats) {
  Board b(side,side);
  if(setup!="-") for(size_t i=0;i<setup.size();i+=2)
    test(b.setStone(decode(setup.at(i),setup.at(i+1)),P_BLACK),"handicap");
  std::vector<Loc> groups;
  Loc chased=Board::NULL_LOC;
  int compared=0, own=0;
  for(int i=0;i<int(seq.size());i+=3) {
    const Player pla=seq[i]=='B'?P_BLACK:P_WHITE;
    const Loc move=decode(seq.at(i+1),seq.at(i+2));
    test(b.isLegal(move,pla,false),"illegal historical move");
    if(pla!=who) {
      groups=touched(b,move,who);
      b.playMoveAssumeLegal(move,pla);
      groups.erase(std::remove_if(groups.begin(),groups.end(),[&](Loc h){return b.colors[h]!=who||b.getNumLiberties(h)!=1;}),groups.end());
      continue;
    }
    ++own;
    Loc next=Board::NULL_LOC;
    if(groups.size()==1) {
      const Loc h=groups[0];
      bool repeated=chased!=Board::NULL_LOC && b.colors[chased]==who && b.chain_head[chased]==b.chain_head[h];
      if(repeated) {
        if(move==last_lib(b,h)) next=move;
      } else {
        const auto e=escape(b,h);
        if(e.valid && move==e.move) next=move;
        if(e.valid && e.full) {
          auto offers=other_rescues(b,h,e.move);
          if((31*game+i/3)%997==0) {
            test(offers==exhaustive(b,h,e.move),"all-board rescue enumeration disagrees");
            ++compared;
          }
          int choice=move==e.move?0:(offers.count(move)?1:2);
          std::array<int,11> row={game,i/3,int(who==P_WHITE),b.getChainSize(h),int(e.local),int(offers.size()),
                 offers.size()==1?offers.begin()->second:0,choice,packed(move),
                 offers.size()==1?packed(offers.begin()->first):-1,packed(e.move)};
          for(int col:row) std::cout<<col<<' ';
          std::cout<<'\n';
        }
      }
    }
    chased=next;
    b.playMoveAssumeLegal(move,pla);
    groups.clear();
  }
  stats<<game<<' '<<own<<' '<<compared<<'\n';
}
}
int main(int argc,char** argv) {
  Board::initHash();
  if(argc!=2) return 2;
  std::ofstream out(argv[1]); if(!out) return 2;
  std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
  int id; char col; std::string setup,seq;
  while(std::cin>>id>>col>>setup>>seq) read_menu(id,col=='B'?P_BLACK:P_WHITE,setup,seq,out);
  return std::cin.eof() && out.good()?0:2;
}
