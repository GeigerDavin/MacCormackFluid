#ifndef SHAREDMATRIX_ZPLANES_HPP
#define SHAREDMATRIX_ZPLANES_HPP

#include <math.h>
#include <iostream>
#include <cstring>
#include <mpi/mpi.h>

#include "SharedMatrix.hpp"

// Mesasge Tags for exchaning Data
#define MASTER_MATRIX_DISTRIBUTE_MSG_TAG (10)
#define MASTER_MATRIX_MERGE_MSG_TAG (11)
#define GHOAST_PLANES_EXCHANGE_MSG_TAG (12)

template<typename T> class SharedMatrix_ZPlanes : public SharedMatrix<T> {
public:
	SharedMatrix_ZPlanes(int mpiRank, int mpiSize, int matX, int matY, int matZ);
	~SharedMatrix_ZPlanes();

	bool doIParticipate() override;

	void startDistributeMasterMatrixAsync() override;
	void waitDistributeMasterMatrixAsyncFinish() override;

	void startMergeMasterMatrixAsync() override;
	void waitMergeMasterMatrixAsyncFinish() override;

	void startExchangeGhostCells() override;
	void waitExchangeGhostCellsFinish() override;

	bool haveGhostCellsXLeft() override;
	bool haveGhostCellsXRight() override;
	bool haveGhostCellsYLeft() override;
	bool haveGhostCellsYRight() override;
	bool haveGhostCellsZLeft() override;
	bool haveGhostCellsZRight() override;

	int getWorkerMatX() override;
	int getWorkerMatY() override;
	int getWorkerMatZ() override;

	int getWorkerOffsetX() override;
	int getWorkerOffsetY() override;
	int getWorkerOffsetZ() override;

private:
	const int _numberOfCellsPerPlane = 0;
	int _myNumberOfPlanes = 0;
	int _myStartPlane = 0;
	int _maxNumberOfPlanesPerWorker = 0;
	int _numberOfParticipantWorker = 0;
	int _restNumberOfPlanesForLastWorker = 0;
	int _lastParticipateWorker = 0;
	int _myNumberOfGhostPlanes = 0;

	MPI_Request* _mergeMasterMatrixRequestsMASTER = nullptr;
	MPI_Status* _mergeMasterMatrixStatusMASTER = nullptr;
	MPI_Request _mergeMasterMatrixRequestsWORKER;
	MPI_Status _mergeMasterMatrixStatusWORKER;

	MPI_Request* _distributeMasterMatrixRequestsMASTER = nullptr;
	MPI_Status* _distributeMasterMatrixStatusMASTER = nullptr;
	MPI_Request _distributeMasterMatrixRequestsWORKER;
	MPI_Status _distributeMasterMatrixStatusWORKER;

	MPI_Request* _exchangeGhostPlaneRequest = nullptr;
	MPI_Status* _exchangeGhostPlaneStatus = nullptr;
};

template<typename T> SharedMatrix_ZPlanes<T>::SharedMatrix_ZPlanes(int mpiRank, int mpiSize, int matX, int matY, int matZ) :
SharedMatrix<T>(mpiRank, mpiSize, matX, matY, matZ), _numberOfCellsPerPlane(matX * matY) {

	// Number of Workers without Master
	int numberOfWorkers = this->_mpiSize - 1;
	// It can not be more workers than _matZ
	_numberOfParticipantWorker = (this->_matZ < numberOfWorkers) ? this->_matZ : numberOfWorkers;
	// Number of Planes for all workers Worker except the last one
	_maxNumberOfPlanesPerWorker = (int)ceil((double) this->_matZ / (double)numberOfWorkers);
	// The last Woker gets the remaining Planes
	_restNumberOfPlanesForLastWorker = this->_matZ - (_maxNumberOfPlanesPerWorker * (_numberOfParticipantWorker - 1));

	// Calculate Number of Planes for THIS Worker
	if (mpiRank == 0) {
		_myNumberOfPlanes = 0;
	}
	else if (_numberOfParticipantWorker == 1 && this->_mpiRank == 1) {
		_myNumberOfPlanes = _maxNumberOfPlanesPerWorker;
	}
	else {
		if (this->_mpiRank < _numberOfParticipantWorker) {
			_myNumberOfPlanes = _maxNumberOfPlanesPerWorker;
		}
		else if (this->_mpiRank == _numberOfParticipantWorker) {
			_myNumberOfPlanes = _restNumberOfPlanesForLastWorker;
		}
	}

	// If THIS Worker participate in this Shared Matrix init some variables
	if (doIParticipate()) {
		_myStartPlane = (this->_mpiRank - 1) * _maxNumberOfPlanesPerWorker;
		_myNumberOfGhostPlanes += haveGhostCellsZLeft() == true ? 1 : 0;
		_myNumberOfGhostPlanes += haveGhostCellsZRight() == true ? 1 : 0;
		this->_workerMatrix = new T[(_myNumberOfGhostPlanes + _myNumberOfPlanes) * _numberOfCellsPerPlane];
		_exchangeGhostPlaneRequest = new MPI_Request[_myNumberOfGhostPlanes * 2];
		_exchangeGhostPlaneStatus = new MPI_Status[_myNumberOfGhostPlanes * 2];
	}
	// Init variables for the Master
	else if (this->_mpiRank == MPI_MASTER) {
		this->_masterMatrix = new T[this->_matX * this->_matY * this->_matZ];
		_mergeMasterMatrixRequestsMASTER = new MPI_Request[_numberOfParticipantWorker];
		_mergeMasterMatrixStatusMASTER = new MPI_Status[_numberOfParticipantWorker];
		_distributeMasterMatrixRequestsMASTER = new MPI_Request[_numberOfParticipantWorker];
		_distributeMasterMatrixStatusMASTER = new MPI_Status[_numberOfParticipantWorker];
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::startDistributeMasterMatrixAsync() {
	if (this->_mpiRank == MPI_MASTER) {
		for (int i = 0; i < _numberOfParticipantWorker; ++i) {
			if ((i + 1) == _numberOfParticipantWorker) {
				// Rcv Data to Last Worker
				MPI_Isend(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
					_restNumberOfPlanesForLastWorker * _numberOfCellsPerPlane, MPI_FLOAT, i + 1,
					MASTER_MATRIX_DISTRIBUTE_MSG_TAG, MPI_COMM_WORLD,
					&(_distributeMasterMatrixRequestsMASTER[i]));
			}
			else {
				// Send Data to all other Workers
				MPI_Isend(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
					_maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane,
					MPI_FLOAT, i + 1,
					MASTER_MATRIX_DISTRIBUTE_MSG_TAG, MPI_COMM_WORLD,
					&(_distributeMasterMatrixRequestsMASTER[i]));
			}
		}
	}
	else {
		if (_myNumberOfPlanes > 0) {
			MPI_Irecv(&(this->_workerMatrix[haveGhostCellsZLeft() ? _numberOfCellsPerPlane : 0]),
				_myNumberOfPlanes * _numberOfCellsPerPlane, MPI_FLOAT,
				MPI_MASTER, MASTER_MATRIX_DISTRIBUTE_MSG_TAG,
				MPI_COMM_WORLD, &_distributeMasterMatrixRequestsWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::waitDistributeMasterMatrixAsyncFinish() {
	if (this->_mpiRank == MPI_MASTER) {
		MPI_Waitall(_numberOfParticipantWorker, _distributeMasterMatrixRequestsMASTER,
			_distributeMasterMatrixStatusMASTER);
	}
	else {
		if (_myNumberOfPlanes > 0) {
			MPI_Wait(&_distributeMasterMatrixRequestsWORKER, &_distributeMasterMatrixStatusWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::startMergeMasterMatrixAsync() {
	if (this->_mpiRank == MPI_MASTER) {
		for (int i = 0; i < _numberOfParticipantWorker; ++i) {

			if ((i + 1) == _numberOfParticipantWorker) {
				// Rcv Data from Last Worker
				MPI_Irecv(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
					_restNumberOfPlanesForLastWorker * _numberOfCellsPerPlane, MPI_FLOAT, i + 1,
					MASTER_MATRIX_MERGE_MSG_TAG, MPI_COMM_WORLD, &(_mergeMasterMatrixRequestsMASTER[i]));
			}
			else {
				// Recv Data from all other Workers
				MPI_Irecv(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
					_maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane,
					MPI_FLOAT, i + 1,
					MASTER_MATRIX_MERGE_MSG_TAG, MPI_COMM_WORLD, &(_mergeMasterMatrixRequestsMASTER[i]));
			}
		}
	}
	else {
		if (_myNumberOfPlanes > 0) {
			MPI_Isend(&(this->_workerMatrix[haveGhostCellsZLeft() ? _numberOfCellsPerPlane : 0]),
				_myNumberOfPlanes * _numberOfCellsPerPlane, MPI_FLOAT,
				MPI_MASTER, MASTER_MATRIX_MERGE_MSG_TAG,
				MPI_COMM_WORLD, &_mergeMasterMatrixRequestsWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::waitMergeMasterMatrixAsyncFinish() {
	if (this->_mpiRank == MPI_MASTER) {
		MPI_Waitall(_numberOfParticipantWorker, _mergeMasterMatrixRequestsMASTER, _mergeMasterMatrixStatusMASTER);
	}
	else {
		if (_myNumberOfPlanes > 0) {
			MPI_Wait(&_mergeMasterMatrixRequestsWORKER, &_mergeMasterMatrixStatusWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::startExchangeGhostCells() {
	if (doIParticipate()) {
		if (!haveGhostCellsZRight() && haveGhostCellsZLeft()) {
			// Ghoast Cells Left Send an Recv
			MPI_Isend(&(this->_workerMatrix[_numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank - 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[0]), _numberOfCellsPerPlane, MPI_FLOAT, this->_mpiRank - 1,
				GHOAST_PLANES_EXCHANGE_MSG_TAG, MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));
		}
		else if (!haveGhostCellsZLeft() && haveGhostCellsZRight()) {
			// Ghoast Cells Right Send an Recv
			MPI_Isend(&(this->_workerMatrix[(_myNumberOfPlanes - 1) * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[_myNumberOfPlanes * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));
		}
		else if (haveGhostCellsZLeft() && haveGhostCellsZRight()) {
			// Ghoast Cells Left Send an Recv
			MPI_Isend(&(this->_workerMatrix[_numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank - 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[0]), _numberOfCellsPerPlane, MPI_FLOAT, this->_mpiRank - 1,
				GHOAST_PLANES_EXCHANGE_MSG_TAG, MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));

			// Ghoast Cells Right Send an Recv
			MPI_Isend(&(this->_workerMatrix[(_myNumberOfPlanes)* _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[2]));

			MPI_Irecv(&(this->_workerMatrix[(1 + _myNumberOfPlanes) * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
				MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
				MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[3]));
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::waitExchangeGhostCellsFinish() {
	if (doIParticipate()) {
		MPI_Waitall(_myNumberOfGhostPlanes * 2, _exchangeGhostPlaneRequest, _exchangeGhostPlaneStatus);
	}
}

template<typename T> bool SharedMatrix_ZPlanes<T>::doIParticipate() {
	return _myNumberOfPlanes > 0;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsXLeft() {
	return false;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsXRight() {
	return false;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsYLeft() {
	return false;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsYRight() {
	return false;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsZLeft() {
	return doIParticipate() ? _myStartPlane > 0 : 0;
}

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsZRight() {
	return (_myStartPlane + _myNumberOfPlanes) < this->_matZ;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerMatX() {
	return doIParticipate() ? this->_matX : 0;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerMatY() {
	return doIParticipate() ? this->_matY : 0;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerMatZ() {
	return doIParticipate() ? _myNumberOfPlanes + _myNumberOfGhostPlanes : 0;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerOffsetX() {
	return 0;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerOffsetY() {
	return 0;
}

template<typename T> int SharedMatrix_ZPlanes<T>::getWorkerOffsetZ() {
	return doIParticipate() ? _myStartPlane : 0;
}

template<typename T> SharedMatrix_ZPlanes<T>::~SharedMatrix_ZPlanes() {

	if (_mergeMasterMatrixRequestsMASTER)
		delete[] _mergeMasterMatrixRequestsMASTER;

	if (_distributeMasterMatrixRequestsMASTER)
		delete[] _distributeMasterMatrixRequestsMASTER;

	if (_distributeMasterMatrixStatusMASTER)
		delete[] _distributeMasterMatrixStatusMASTER;

	if (_mergeMasterMatrixStatusMASTER)
		delete[] _mergeMasterMatrixStatusMASTER;

	if (_exchangeGhostPlaneRequest)
		delete[] _exchangeGhostPlaneRequest;

	if (_exchangeGhostPlaneStatus)
		delete[] _exchangeGhostPlaneStatus;
}

#endif //SHAREDMATRIX_ZPLANES_HPP
