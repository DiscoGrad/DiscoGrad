for y_axis in y dydx0; do
  for paths in 4 16 64 256; do
    echo "generating combined_stddevs_${paths}_paths_$y_axis.pdf"
    #./plot.py results/*stddev=0.002*dgsi_num_paths=$paths*mode=[03]* results/*stddev=0.002*stoch*samples=1000* x0 $y_axis || exit
    ./plot.py --norm results/*stddev=0.002*dgsi_num_paths=$paths*mode=[03]* x0 $y_axis || exit
  
    mv plot.pdf combined_stddevs_${paths}_paths_$y_axis.pdf
  
    for stddev in 0.008 0.032 0.128 0.512 2.048; do
      #./plot.py results/*stddev=$stddev*dgsi_num_paths=$paths*mode=[03]* results/*stddev=$stddev*stoch*samples=1000* x0 $y_axis || exit
      ./plot.py --norm results/*stddev=$stddev*dgsi_num_paths=$paths*mode=[03]* x0 $y_axis || exit
      pdftk combined_stddevs_${paths}_paths_$y_axis.pdf plot.pdf output out.pdf
      mv out.pdf combined_stddevs_${paths}_paths_$y_axis.pdf
    done
  done

  for stddev in 0.002 0.008 0.032 0.128 0.512 2.048; do
    echo "generating combined_paths_stddev_${stddev}_$y_axis.pdf"
    #./plot.py results/*stddev=$stddev*dgsi_num_paths=4*mode=[03]* results/*stddev=$stddev*stoch* x0 $y_axis || exit
    ./plot.py results/*stddev=$stddev*dgsi_num_paths=4*mode=[03]* x0 $y_axis || exit
  
    mv plot.pdf combined_paths_stddev_${stddev}_$y_axis.pdf
  
    for paths in 16 64 256; do
      #./plot.py results/*stddev=$stddev*dgsi_num_paths=$paths*mode=[03]* results/*stddev=$stddev*stoch* x0 $y_axis || exit
      ./plot.py results/*stddev=$stddev*dgsi_num_paths=$paths*mode=[0]* x0 $y_axis || exit
      pdftk combined_paths_stddev_${stddev}_$y_axis.pdf plot.pdf output out.pdf
      mv out.pdf combined_paths_stddev_${stddev}_$y_axis.pdf
    done
  done
done
