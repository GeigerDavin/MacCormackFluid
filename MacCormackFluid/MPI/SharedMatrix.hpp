#ifndef SHAREDMATRIX_HPP
#define SHAREDMATRIX_HPP

#define MPI_MASTER (0)

template<typename T> class SharedMatrix {
public:
	SharedMatrix(int mpiRank, int mpiSize, int matX, int matY, int matZ) :
		_mpiRank(mpiRank), _mpiSize(mpiSize), _matX(matX), _matY(matY), _matZ(matZ) {}
	virtual ~SharedMatrix() {};

	virtual bool doIParticipate() = 0;

	virtual void startDistributeMasterMatrixAsync() = 0;
	virtual void waitDistributeMasterMatrixAsyncFinish() = 0;

	virtual void startMergeMasterMatrixAsync() = 0;
	virtual void waitMergeMasterMatrixAsyncFinish() = 0;

	virtual void startExchangeGhostCells() = 0;
	virtual void waitExchangeGhostCellsFinish() = 0;

	virtual bool haveGhostCellsXLeft() = 0;
	virtual bool haveGhostCellsX() = 0;
	virtual bool haveGhostCellsYLeft() = 0;
	virtual bool haveGhostCellsYRight() = 0;
	virtual bool haveGhostCellsZLeft() = 0;
	virtual bool haveGhostCellsZRight() = 0;

	virtual T* getWorkerMatrix();
	virtual int getWorkerMatX() = 0;
	virtual int getWorkerMatY() = 0;
	virtual int getWorkerMatZ() = 0;

	virtual int getWorkerOffsetX() = 0;
	virtual int getWorkerOffsetY() = 0;
	virtual int getWorkerOffsetZ() = 0;

	virtual T* getMasterMatrix();
	virtual int getMasterMatX();
	virtual int getMasterMatY();
	virtual int getMasterMatZ();

protected:
	const int _mpiRank = 0;
	const int _mpiSize = 0;

	const int _matX = 0;
	const int _matY = 0;
	const int _matZ = 0;

public:
	T* _workerMatrix = nullptr;
	T* _masterMatrix = nullptr;
};

template<typename T> T* SharedMatrix<T>::getWorkerMatrix() {
	return _workerMatrix;
}

template<typename T> T* SharedMatrix<T>::getMasterMatrix() {
	return _masterMatrix;
}

template<typename T> int SharedMatrix<T>::getMasterMatX() {
	return _matX;
}

template<typename T> int SharedMatrix<T>::getMasterMatY() {
	return _matY;
}

template<typename T> int SharedMatrix<T>::getMasterMatZ() {
	return _matZ;
}

#endif //SHAREDMATRIX_HPP
