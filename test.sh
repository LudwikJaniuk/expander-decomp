echo "Expect: diff 0; conductance 0.010989; expansion 0.1; case 2"
/home/janiuk/Code/individuellt/cmake-build-debug/a.out -f graphs/barbell10-10.graph -r 0 -s --H_phi=0.4 --G_phi=0.1 --vol 0.1 |  grep -E 'CASE|diff|final_expa|final_condu|on rou'
echo ""
echo "Expect: diff 0; conductance 0.000101; expansion 0.01; case 2"
/home/janiuk/Code/individuellt/cmake-build-debug/a.out -f graphs/barbell100-100.graph -r 0 -s --H_phi=0.4 --G_phi=0.1 --vol 0.1 |  grep -E 'CASE|diff|final_expa|final_condu|on rou'
echo ""
echo "Expect: factor 1; conductance 999; expansion 0; case 1"
/home/janiuk/Code/individuellt/cmake-build-debug/a.out -f graphs/complete100.graph -r 0 -s --H_phi=0.4 --G_phi=0.1 --vol 0.1 |  grep -E 'CASE|diff|final_expa|final_condu|on rou'
echo ""
echo "Expect: diff 14; conductance 0.235294; expansion 4; on round 2 (fluke); case 1"
/home/janiuk/Code/individuellt/cmake-build-debug/a.out -f graphs/expander16.graph -r 0 -s --H_phi=0.4 --G_phi=0.1 --vol 0.1 |  grep -E 'CASE|diff|final_expa|final_condu|on rou'


