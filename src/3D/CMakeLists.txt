cmake_minimum_required(VERSION 2.8)

file(GLOB CUDA_FILES "*.cu")
message("CU files: " ${CUDA_FILES})

cuda_add_executable(HDoG3D_main
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_main.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)

cuda_add_executable(HDoG3D_NeuN
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN.h
)

cuda_add_executable(HDoG3D_NeuN_ver2
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver2.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver2.h
)

cuda_add_executable(HDoG3D_NeuN_ver3
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3.h
)
cuda_add_executable(HDoG3D_NeuN_ver3_CCL_save
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3_CCL_save.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3.h
)
cuda_add_executable(HDoG3D_NeuN_ver3_Rank_simple
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3_Rank_simple.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3.h
)

cuda_add_executable(HDoG3D_NeuN_ver3_Rank_simple_3_color
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3_Rank_simple_3_color.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3.h
)

cuda_add_executable(HDoG3D_NeuN_ver3_Rank
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3_Rank.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3.h
)

cuda_add_executable(HDoG3D_NeuN_ver3_2color
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_ver3_2color.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN_ver3_2color.h
)

cuda_add_executable(HDoG3D_NeuN_only
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_NeuN_only.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils_NeuN.h
)

#cuda_add_executable(HDoG3D_AAV
#        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_AAV.cpp
#        ${CUDA_FILES}
#        ${CMAKE_CURRENT_SOURCE_DIR}/utils_AAV.h
#)

#cuda_add_executable(HDoG3D_KTR
#        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_KTR.cpp
#        ${CUDA_FILES}
#        ${CMAKE_CURRENT_SOURCE_DIR}/utils_KTR.h
#)


install(TARGETS HDoG3D_main RUNTIME DESTINATION ".")
install(TARGETS HDoG3D_NeuN RUNTIME DESTINATION ".")
#install(TARGETS HDoG3D_KTR RUNTIME DESTINATION ".")
