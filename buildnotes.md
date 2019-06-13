Note that in case of 
        'target/debug/concurrentgraph: error while loading shared libraries: libconcgcc.so: cannot open shared object file: No such file or directory'
    occurring after `cargo run`, use `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/windows/Users/Wesley/dev/concurrentgraph/concurrentgraph/lib`(wes-linux partition) or `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/c/Users/Wesley/dev/concurrentgraph/concurrentgraph/lib`(wes-wsl partition)

May also need:
    `sudo ldconfig`