cmake_minimum_required(VERSION 3.20)

project(FaceWebChain LANGUAGES NONE)

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

set(FACE_IMAGE "${CMAKE_SOURCE_DIR}/input" CACHE PATH "Authorized input image or folder passed to the run target")
set(FACE_THRESHOLD "0.50" CACHE STRING "Face distance threshold passed to the run target")
set(FACE_UPSAMPLE_TIMES "0" CACHE STRING "Face detection upsample count passed to the run target")
set(FACE_OUTPUT "artifacts" CACHE PATH "Output directory passed to the run target")

set(FACE_VENV_DIR "${CMAKE_BINARY_DIR}/.venv")
if(WIN32)
    set(FACE_VENV_PYTHON "${FACE_VENV_DIR}/Scripts/python.exe")
else()
    set(FACE_VENV_PYTHON "${FACE_VENV_DIR}/bin/python")
endif()

set(FACE_DEPENDENCY_STAMP "${CMAKE_BINARY_DIR}/.dependencies-installed")
add_custom_command(
    OUTPUT "${FACE_DEPENDENCY_STAMP}"
    COMMAND "${Python3_EXECUTABLE}" -m venv "${FACE_VENV_DIR}"
    COMMAND "${FACE_VENV_PYTHON}" -m pip install --upgrade pip
    COMMAND "${FACE_VENV_PYTHON}" -m pip install -r "${CMAKE_SOURCE_DIR}/requirements.txt"
    COMMAND "${FACE_VENV_PYTHON}" -m pip install -r "${CMAKE_SOURCE_DIR}/requirements-dev.txt"
    COMMAND "${CMAKE_COMMAND}" -E touch "${FACE_DEPENDENCY_STAMP}"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/requirements.txt"
        "${CMAKE_SOURCE_DIR}/requirements-dev.txt"
    COMMENT "Creating the project virtual environment and installing dependencies"
    VERBATIM
)

add_custom_target(setup DEPENDS "${FACE_DEPENDENCY_STAMP}")

add_custom_target(test
    COMMAND "${FACE_VENV_PYTHON}" -m pytest -q
    DEPENDS setup
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running the offline test suite"
    VERBATIM
)

add_custom_target(deploy-contract
    COMMAND "${FACE_VENV_PYTHON}" -m app.deploy_contract
    DEPENDS setup
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Deploying MatchRegistry.sol to Sepolia"
    VERBATIM
)

add_custom_target(run
    COMMAND "${FACE_VENV_PYTHON}" -m app.cli run
        --image "${FACE_IMAGE}"
        --threshold "${FACE_THRESHOLD}"
        --upsample-times "${FACE_UPSAMPLE_TIMES}"
        --output "${FACE_OUTPUT}"
    DEPENDS setup
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running the FaceWebChain pipeline"
    VERBATIM
)

message(STATUS "FaceWebChain configured with Python: ${Python3_EXECUTABLE}")
message(STATUS "Build targets: setup, test, deploy-contract, run")