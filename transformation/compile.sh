#!/bin/bash

compile() {
    clang++ -g -w -c -I/usr/lib/llvm-13/include -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS  -Wall -Wextra -I/usr/lib/llvm-13/include -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti $basename.cpp &&\

    clang++ -g -w -o $basename -Wall -Wextra -I/usr/lib/llvm-13/include -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti -L/usr/lib/llvm-13/lib $basename.o -Wl,--start-group -lclang -lclangFrontend -lclangDriver -lclangSerialization -lclangParse -lclangSema -lclangAnalysis -lclangEdit -lclangAST -lclangLex -lclangBasic -lclangTooling -lclangRewrite -lclangRewriteFrontend -Wl,--end-group -lLLVM-13
}

fc37_compile() {
    clang++ -g -w -c -I/usr/lib/llvm-13/include -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS  -Wall -Wextra -I/usr/lib/llvm-13/include -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti $basename.cpp &&\

    clang++ -g -w -o $basename -Wall -Wextra -I/usr/lib/llvm-13/include -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti -L/usr/lib/llvm-13/lib $basename.o -Wl,--start-group -lclang-cpp -Wl,--end-group -lLLVM-16
}

fc38_compile() {
    clang++ -g -w -c -I/usr/include/clang -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS  -Wall -Wextra -I/usr/include/clang -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti $basename.cpp &&\

    clang++ -g -w -o $basename -Wall -Wextra -I/usr/include/clang -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti -L/usr/include/clang $basename.o -Wl,--start-group -lclang-cpp -Wl,--end-group -lLLVM-16
}

arch_compile() {
    clang++ -g -w -c -I/usr/include/clang -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS  -Wall -Wextra -I/usr/include/clang -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti $basename.cpp &&\

    clang++ -g -w -o $basename -Wall -Wextra -I/usr/include/clang -std=c++20   -fno-exceptions -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti -L/usr/include/clang $basename.o -Wl,--start-group -lclang-cpp -Wl,--end-group -lLLVM-15
}


for basename in normalize smooth_dgo; do
# if the kernel is fedora or arch linux, use different linking args
kernel_name="$(uname -r)"
  if [[ $kernel_name == *"fc37"* ]]; then  
    echo "Compiling $basename under Fedora 37"
    fc37_compile
  elif [[ $kernel_name == *"fc38"* ]]; then
    echo "Compiling $basename under Fedora 38"
    fc38_compile
  elif [[ $kernel_name == *"arch"* ]]; then
    echo "Compiling $basename under Arch"
    arch_compile
  else
    echo "Compiling $basename"
    compile
  fi
done
