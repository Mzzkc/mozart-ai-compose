#!/bin/bash
# DJ GestAIt — AI Set
# Prime number phasing, euclidean geometry, phase lock and scatter
# The set that made the composer say "holy shit"

TARGET="${1:-workspaces/dj-gestait-system-lab/run/live/current.strudel}"

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  note("c2").sound("sine").gain(sine.range(0, 0.5).slow(7)),
  note("g2").sound("sine").gain(sine.range(0, 0.4).slow(11)),
  note("eb3").sound("sine").gain(sine.range(0, 0.3).slow(13))
)
S
sleep 12

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  note("c2").sound("sine").gain(sine.range(0, 0.5).slow(7)),
  note("g2").sound("sine").gain(sine.range(0, 0.4).slow(11)),
  note("eb3").sound("sine").gain(sine.range(0, 0.3).slow(13)),
  s("bd(3,8)").gain(sine.range(0, 0.6).slow(17)).lpf(sine.range(60, 300).slow(19))
)
S
sleep 10

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(3,8)").gain(0.7).lpf(sine.range(200, 800).slow(19)),
  s("hh(5,8)").gain(sine.range(0.05, 0.2).slow(7)).hpf(6000).pan(sine.range(0.1, 0.9).slow(11)),
  s("cp(2,8)").gain(0.25).room(0.6).delay(0.3).delaytime(sine.range(0.125, 0.375).slow(23)),
  note("<c2 c2 g1 c2 c2 eb2 c2>").sound("sawtooth").lpf(sine.range(100, 500).slow(29)).gain(0.45)
)
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(3,8)").gain(0.75).lpf(sine.range(300, 1200).slow(19)),
  s("hh(5,8)").gain(sine.range(0.08, 0.25).slow(7)).hpf(5000).pan(sine.range(0.1, 0.9).slow(11)),
  s("cp(2,8)").gain(0.3).room(0.5).delay(0.25).delaytime(sine.range(0.125, 0.375).slow(23)),
  s("oh(1,8)").gain(0.15).hpf(3000).room(0.4),
  note("<c2 c2 g1 c2 c2 eb2 c2>").sound("sawtooth").lpf(sine.range(150, 800).slow(29)).gain(0.5),
  s("rim(3,8,1)").gain(0.2).delay(0.2).delaytime(0.166).pan(sine.range(0.3, 0.7).slow(13))
)
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(5,8)").gain(0.8).lpf(sine.range(400, 2000).slow(17)),
  s("hh(7,16)").gain(sine.range(0.1, 0.3).slow(7)).hpf(4000).pan(sine.range(0.05, 0.95).slow(11)),
  s("cp(3,8)").gain(0.35).room(0.4).sometimesBy(0.2, x => x.speed(1.5)),
  s("oh(2,8)").gain(0.2).hpf(3000),
  note("c2 eb2 g1 bb0 c2 eb2 g1 c2").sound("sawtooth").lpf(sine.range(200, 2500).slow(23)).gain(0.55).resonance(sine.range(1, 12).slow(31)),
  note("<[c3,eb3,g3] [bb2,d3,f3] [eb3,g3,bb3]>").sound("triangle").gain(sine.range(0, 0.2).slow(37)).room(0.9).size(0.9).lpf(sine.range(300, 1500).slow(41)),
  s("rim(5,16,3)").gain(0.15).delay(0.15).delaytime(sine.range(0.1, 0.25).slow(13))
)
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(7,16)").gain(0.85).lpf(2000),
  s("hh(11,16)").gain(0.25).hpf(3000).pan(sine.range(0, 1).slow(3)),
  s("cp(5,16)").gain(0.35).room(0.3).every(3, x => x.fast(2)),
  s("oh(3,8)").gain(0.22).hpf(2000),
  s("rim(7,16,2)").gain(0.2).delay(0.2).delaytime(0.125),
  note("c2 eb2 g1 bb0 c2 g1 eb2 bb0").sound("sawtooth").lpf(sine.range(300, 4000).slow(11)).gain(0.55).resonance(12),
  note("[c3,eb3,g3,bb3,d4]").sound("sine").gain(0.18).room(0.9).lpf(sine.range(500, 3000).slow(17)),
  s("bd(3,8,1)").gain(0.3).hpf(1000).speed(2)
)
S
sleep 6

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
note("c4 ~ ~ ~ ~ eb4 ~ ~ ~ ~ g4 ~ ~ ~ ~ ~").sound("triangle")
  .room(0.95).size(0.95)
  .delay(0.8).delaytime(0.5).delayfeedback(0.75)
  .gain(0.3)
  .lpf(sine.range(400, 2000).slow(29))
  .pan(sine.range(0.2, 0.8).slow(7))
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(3,8,2)").gain(0.7).lpf(sine.range(200, 600).slow(13)),
  note("c4 ~ ~ ~ ~ eb4 ~ ~ ~ ~ g4 ~ ~ ~ ~ ~").sound("triangle")
    .room(0.9).gain(sine.range(0.2, 0).slow(16))
    .delay(0.6).delaytime(0.5).delayfeedback(0.6),
  s("hh(5,16,1)").gain(sine.range(0, 0.2).slow(11)).hpf(6000),
  note("g1 ~ c2 ~").sound("sawtooth").lpf(sine.range(80, 400).slow(17)).gain(0.4)
)
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(4,8)").gain(0.8).lpf(sine.range(300, 1500).slow(13)),
  s("hh(7,16,2)").gain(sine.range(0.08, 0.28).slow(7)).hpf(5000).pan(sine.range(0.15, 0.85).slow(11)),
  s("cp(2,8,1)").gain(0.35).room(0.4).delay(0.2).delaytime(0.33).delayfeedback(0.4),
  s("oh(1,4)").gain(0.18).hpf(3000),
  note("g1 c2 eb2 <c2 bb0> g1 c2 eb2 c2").sound("sawtooth").lpf(sine.range(200, 2000).slow(19)).gain(0.55).resonance(sine.range(2, 10).slow(23)),
  note("<[g3,bb3,d4] [c3,eb3,g3] [eb3,g3,bb3]>").sound("sine").gain(sine.range(0.05, 0.2).slow(29)).room(0.85).lpf(sine.range(400, 1800).slow(31))
)
S
sleep 10

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(5,8)").gain(0.85).lpf(sine.range(400, 2500).slow(7)),
  s("hh(9,16)").gain(sine.range(0.1, 0.35).slow(7)).hpf(4000).pan(sine.range(0, 1).slow(7)),
  s("cp(3,8)").gain(0.4).room(0.3).delay(0.15).delaytime(sine.range(0.125, 0.25).slow(7)),
  s("oh(2,8)").gain(0.22).hpf(3000).room(0.3),
  s("ride(1,4)").gain(0.1).hpf(7000),
  note("c2 eb2 g1 bb0 c2 g1 eb2 c2").sound("sawtooth").lpf(sine.range(300, 4000).slow(7)).gain(0.6).resonance(sine.range(4, 14).slow(7)),
  note("[c3,eb3,g3,bb3,d4]").sound("sine").gain(sine.range(0.08, 0.25).slow(7)).room(0.9).size(0.9).lpf(sine.range(500, 2500).slow(7)),
  s("rim(5,16)").gain(0.2).delay(0.1).delaytime(0.143)
)
S
sleep 8

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd*4").gain(0.9),
  s("hh*16").gain(0.3).hpf(4000),
  s("~ cp ~ cp").gain(0.45),
  s("~ oh ~ oh").gain(0.25),
  note("c2 c2 c2 c2").sound("sawtooth").lpf(4000).gain(0.6),
  note("[c3,eb3,g3,bb3,d4]").sound("sine").gain(0.25).room(0.9)
)
S
sleep 4

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  s("bd(3,8)").gain(sine.range(0.8, 0.3).slow(13)),
  s("hh(5,8)").gain(sine.range(0.25, 0.05).slow(17)).hpf(5000),
  note("c2").sound("sawtooth").lpf(sine.range(800, 100).slow(19)).gain(sine.range(0.5, 0.1).slow(23)),
  note("[c3,eb3,g3]").sound("sine").gain(sine.range(0.2, 0).slow(29)).room(0.95)
)
S
sleep 10

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
stack(
  note("c2").sound("sine").gain(sine.range(0.4, 0).slow(11)),
  note("g2").sound("sine").gain(sine.range(0.3, 0).slow(13)),
  note("eb3").sound("sine").gain(sine.range(0.2, 0).slow(17))
)
S
sleep 12

cat > "$TARGET" << 'S'
setcps(133 / 60 / 4)
silence
S

echo "Set complete."
