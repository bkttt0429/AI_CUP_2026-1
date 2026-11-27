NR>1 { n=gsub(/;[BW]\[/,""); t++; if(n>=250)c++; if(n>=300)d++ }
END  { printf "%-14s total=%d  >=250 moves: %d (%.1f%%)  >=300: %d (%.1f%%)\n", FILENAME,t,c,100*c/t,d,100*d/t }
