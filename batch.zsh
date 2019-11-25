#  ./cmake-build-debug/a.out -f graphs/NAME.graph -p partitions/5/NAME.2.ptn -r 5 -s -S -o out/NAME.ptn > out/NAME.out; echo NAME;
#./cmake-build-debug/a.out -f graphs/144.graph -p partitions/5/144.2.ptn -r 3 -s -o out/144.ptn > out/144.out
#echo 144


# 144

#./cmake-build-debug/a.out -f graphs/3elt.graph -p partitions/5/3elt.2.ptn -r 5 -s -o out/3elt.ptn > out/3elt.out; echo 3elt;


#./cmake-build-debug/a.out -f graphs/4elt.graph -p partitions/5/4elt.2.ptn -r 5 -s -S -o out/4elt.ptn > out/4elt.out; echo 4elt;

for NAME in 144 3elt 4elt 598a add20 add32 auto bcsstk29 bcsstk30 bcsstk31 bcsstk32 bcsstk33 brack2 crack cs4 cti data fe_4elt2 fe_body fe_ocean fe_pwt fe_rotor fe_sphere fe_tooth finan512 m14b memplus t60k uk vibrobox wave whitaker3 wing wing_nodal; do 
	./cmake-build-debug/a.out -f graphs/$NAME.graph -p partitions/5/$NAME.2.ptn -r 25 -s -o out/$NAME.ptn > out/$NAME.out; 
	echo $NAME; 
done;
#) do; ./cmake-build-debug/a.out -f graphs/$NAME.graph -p partitions/5/$NAME.2.ptn -r 5 -s -S -o out/$NAME.ptn > out/$NAME.out; echo $NAME; done;

# 598a
# add20
# add32
# auto
# bcsstk29
# bcsstk30
# bcsstk31
# bcsstk32
# bcsstk33
# brack2
# crack
# cs4
# cti
# data
# fe_4elt2
# fe_body
# fe_ocean
# fe_pwt
# fe_rotor
# fe_sphere
# fe_tooth
# finan512
# m14b
# memplus
# t60k
# uk
# vibrobox
# wave
# whitaker3
# wing
# wing_nodal
