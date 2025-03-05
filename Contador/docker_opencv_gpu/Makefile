# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: actual_build build_prep

# Adaptado de
# DATAMACHINES.IO
# https://github.com/datamachines/cuda_tensorflow_opencv


# Release to match data of Dockerfile and follow YYYYMMDD pattern
# DATAMACHINES CTO_RELEASE 20200615

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# According to https://hub.docker.com/r/nvidia/cuda/
# CUDA 11 came out in May 2020
# TF1 needs CUDA 10.1
# TF2 needs CUDA 10.2
# Table below shows driver/CUDA support; for example the 10.2 container needs at least driver 440.33
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
STABLE_CUDA9=9.2
STABLE_CUDA10=10.2
STABLE_CUDA101=10.1
# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
# Skipping Tegra, Jetson, ... from this list
# Keeping Pascal and above
DNN_ARCH_CUDA9=6.1#6.0,6.1
DNN_ARCH_CUDA10=6.1#6.0,6.1,7.0,7.5

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.10
STABLE_OPENCV4=4.3.0

# According to https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md
STABLE_TF1=1.15.3
STABLE_TF2=2.2.0,
STABLE_TF21=2.1.0

# Information for build

##### CUDA _ Tensorflow _ OpenCV
CTO_BUILDALL =cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA101}_${STABLE_TF21}_${STABLE_OPENCV4}

##### CuDNN _ Tensorflow _ OpenCV
DTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA101}_${STABLE_TF21}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "** Imagens de Docker disponíveis (make targets):"
	@echo "  cuda_tensorflow_opencv: "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cudnn_tensorflow_opencv: "; echo -n "      "; echo ${DTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** examplo de build:
	@echo "  make cudnn_tensorflow_opencv-9.2_1.15.3_4.3.0"
	@echo "  make cudnn_tensorflow_opencv-10.2_1.15.3_4.3.0"
	@echo ""
	@echo "Nota: TensorFlow GPU support can only be compiled for CuDNN containers"


${CTO_BUILDALL}:
	@CUDX="cuda" CUDX_FROM="" CUDX_COMP="" BTARG="$@" make build_prep

${DTO_BUILDALL}:
	@CUDX="cudnn" CUDX_FROM="-cudnn7" CUDX_COMP="-DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON" BTARG="$@" make build_prep
# CUDA_ARCH_BIN is set in build_prep now 

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/\.\d+/.0/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

	@$(eval CTO_TMP="cuda-npp-${CTO_CUDA_VERSION} cuda-cublas-${CTO_CUDA_PRIMEVERSION} cuda-cufft-${CTO_CUDA_VERSION} cuda-libraries-${CTO_CUDA_VERSION} cuda-npp-dev-${CTO_CUDA_VERSION} cuda-cublas-dev-${CTO_CUDA_PRIMEVERSION} cuda-cufft-dev-${CTO_CUDA_VERSION} cuda-libraries-dev-${CTO_CUDA_VERSION}")
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@$(eval CTO_DNN_ARCH=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${DNN_ARCH_CUDA9}"; else echo "${DNN_ARCH_CUDA10}"; fi))
	@$(eval CUDX_COMP=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${CTO_DNN_ARCH}"; else echo "${CUDX_COMP}"; fi))

	@$(eval CTO_FROM=$(shell if [ ${CTO_SC} == 1 ]; then echo "ubuntu:18.04"; else echo "nvidia/cuda:${CTO_CUDA_VERSION}${CUDX_FROM}-devel-ubuntu18.04"; fi))

	@$(eval CTO_TMP="-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 ${CUDX_COMP} -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_cublas_LIBRARY=cublas -DCUDA_cufft_LIBRARY=cufft -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppidei_LIBRARY=nppidei -DCUDA_nppif_LIBRARY=nppif -DCUDA_nppig_LIBRARY=nppig -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppist_LIBRARY=nppist -DCUDA_nppisu_LIBRARY=nppisu -DCUDA_nppitc_LIBRARY=nppitc -DCUDA_npps_LIBRARY=npps -DCUDA_nppc_LIBRARY=nppc -DCUDA_nppial_LIBRARY=nppial -DCUDA_nppicc_LIBRARY=nppicc -D CUDA_nppicom_LIBRARY=nppicom")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@echo ""; echo ""
	@echo "[*****] Iniciar build de imagem labcog/${CTO_NAME}:${CTO_TAG}"

	@if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_VERSION=${CTO_TENSORFLOW_VERSION} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" CTO_TF_CUDNN="${CTO_TF_CUDNN}" CTO_TF_OPT="${CTO_TF_OPT}" CTO_TF_KERAS="${CTO_TF_KERAS}" CTO_DNN_ARCH="${CTO_DNN_ARCH}" make actual_build; fi


actual_build:
	@echo "Pressione Ctl+c em até 10 seconds para cancelar"
	@echo "  CTO_FROM               : ${CTO_FROM}"
	@for i in 10 9 8 7 6 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	docker build \
	  --build-arg CTO_FROM=${CTO_FROM} \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --build-arg CTO_CUDA_APT="${CTO_CUDA_APT}" \
	  --build-arg CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" \
	  --build-arg CTO_DNN_ARCH="${CTO_DNN_ARCH}" \
	  --tag="labcog/${CTO_NAME}:${CTO_TAG}" \
	  . | tee build_${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}" 
	@mv build_${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@docker run --rm labcog/${CTO_NAME}:${CTO_TAG} opencv_version -v >> ${CTO_NAME}-${CTO_TAG}.txt
	@docker run --rm labcog/${CTO_NAME}:${CTO_TAG} /tmp/tf_info.sh > ${CTO_NAME}-${CTO_TAG}.txt

clean:
	rm -f *.log.temp

allclean:
	@make clean
	rm -f *.log