echo "before"
ulimit -Sv 100
./cmake-build-debug/a.out --ignore-multi -f graphs/fe_4elt2.graph -p partitions/5/fe_4elt2.2.ptn -s -r 1
echo "after"
