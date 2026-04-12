# Helper: add an exercise with its test and benchmark
function(add_exercise NAME)
    add_executable(test_${NAME} ${NAME}/test_${NAME}.cu)
    target_link_libraries(test_${NAME} PRIVATE GTest::gtest_main)
    add_test(NAME test_${NAME} COMMAND test_${NAME})

    set(BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/bench_${NAME}.cu")
    if(EXISTS "${BENCH_SRC}")
        add_executable(bench_${NAME} ${NAME}/bench_${NAME}.cu)
        target_link_libraries(bench_${NAME} PRIVATE benchmark::benchmark_main)
    endif()
endfunction()

# --- Exercises ---
add_exercise(vecadd)
