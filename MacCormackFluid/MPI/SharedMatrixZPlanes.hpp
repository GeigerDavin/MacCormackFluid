#ifndef SHAREDMATRIX_ZPLANES_HPP
#define SHAREDMATRIX_ZPLANES_HPP

#define MASTER_MATRIX_DISTRIBUTE_MSG_TAG (10)
#define MASTER_MATRIX_MERGE_MSG_TAG (11)
#define GHOAST_PLANES_EXCHANGE_MSG_TAG (12)

#include <math.h>
#include <iostream>
#include <cstring>
#include <mpi/mpi.h>

#include "SharedMatrix.hpp"
#define MPI_MASTER (0)

template<typename T> class SharedMatrix_ZPlanes: public SharedMatrix<T> {
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
	bool haveGhostCellsX() override;
	bool haveGhostCellsYLeft() override;
	bool haveGhostCellsYRight() override;
	bool haveGhostCellsZLeft() override;
	bool haveGhostCellsZRight() override;

	int getWorkerMatX();
	int getWorkerMatY();
	int getWorkerMatZ();
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

template<typename T> bool SharedMatrix_ZPlanes<T>::doIParticipate() {
	return _myNumberOfPlanes > 0;
}

template<typename T> void SharedMatrix_ZPlanes<T>::startDistributeMasterMatrixAsync() {
	if (this->_mpiRank == MPI_MASTER) {
		for (int i = 0; i < _numberOfParticipantWorker; ++i) {
			if ((i + 1) == _numberOfParticipantWorker) {
				MPI_Isend(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
						_restNumberOfPlanesForLastWorker * _numberOfCellsPerPlane, MPI_FLOAT, i+1,
						MASTER_MATRIX_DISTRIBUTE_MSG_TAG, MPI_COMM_WORLD,
						&(_distributeMasterMatrixRequestsMASTER[i]));
			} else {
				MPI_Isend(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
						_maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane,
						MPI_FLOAT, i+1,
						MASTER_MATRIX_DISTRIBUTE_MSG_TAG, MPI_COMM_WORLD,
						&(_distributeMasterMatrixRequestsMASTER[i]));
			}

		}
	} else {
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
		//std::memcpy(&(this->_myMatrix[0]), &(this->_masterMatrix[(haveGhostCellsZLeft() ? _numberOfCellsPerPlane : 0)]),
			//	sizeof(T) * _numberOfCellsPerPlane * _myNumberOfPlanes);
		MPI_Waitall(_numberOfParticipantWorker, _distributeMasterMatrixRequestsMASTER,
				_distributeMasterMatrixStatusMASTER);
	} else {
		if (_myNumberOfPlanes > 0) {
			MPI_Wait(&_distributeMasterMatrixRequestsWORKER, &_distributeMasterMatrixStatusWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::startMergeMasterMatrixAsync() {
	if (this->_mpiRank == MPI_MASTER) {
		for (int i = 0; i < _numberOfParticipantWorker; ++i) {

			if ((i + 1) == _numberOfParticipantWorker) {
				MPI_Irecv(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
						_restNumberOfPlanesForLastWorker * _numberOfCellsPerPlane, MPI_FLOAT, i+1,
						MASTER_MATRIX_MERGE_MSG_TAG, MPI_COMM_WORLD, &(_mergeMasterMatrixRequestsMASTER[i]));
			} else {
				MPI_Irecv(&(this->_masterMatrix[i * _maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane]),
						_maxNumberOfPlanesPerWorker * _numberOfCellsPerPlane,
						MPI_FLOAT, i+1,
						MASTER_MATRIX_MERGE_MSG_TAG, MPI_COMM_WORLD, &(_mergeMasterMatrixRequestsMASTER[i]));
			}

		}
	} else {
		if (_myNumberOfPlanes > 0 && this->_mpiRank == 1) {
			MPI_Isend(&(this->_workerMatrix[haveGhostCellsZLeft() ? _numberOfCellsPerPlane : 0]),
					_myNumberOfPlanes * _numberOfCellsPerPlane, MPI_FLOAT,
					MPI_MASTER, MASTER_MATRIX_MERGE_MSG_TAG,
					MPI_COMM_WORLD, &_mergeMasterMatrixRequestsWORKER);
		}
	}
}
template<typename T> void SharedMatrix_ZPlanes<T>::waitMergeMasterMatrixAsyncFinish() {
	if (this->_mpiRank == MPI_MASTER) {
		//memcpy(&(this->_masterMatrix[0]), &(this->_myMatrix[0]),
				//sizeof(T) * _numberOfCellsPerPlane * _myNumberOfPlanes);

		MPI_Waitall(_numberOfParticipantWorker, _mergeMasterMatrixRequestsMASTER, _mergeMasterMatrixStatusMASTER);

	} else if (_myNumberOfPlanes > 0 && this->_mpiRank == 1){
		if (_myNumberOfPlanes > 0) {
			MPI_Wait(&_mergeMasterMatrixRequestsWORKER, &_mergeMasterMatrixStatusWORKER);
		}
	}
}

template<typename T> void SharedMatrix_ZPlanes<T>::startExchangeGhostCells() {
	if (doIParticipate()) {
		if (!haveGhostCellsZRight() && haveGhostCellsZLeft()) {
			MPI_Isend(&(this->_workerMatrix[_numberOfCellsPerPlane]), _numberOfCellsPerPlane,
			MPI_FLOAT, this->_mpiRank - 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
			MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[0]), _numberOfCellsPerPlane, MPI_FLOAT, this->_mpiRank - 1,
			GHOAST_PLANES_EXCHANGE_MSG_TAG, MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));
		} else if (!haveGhostCellsZLeft() && haveGhostCellsZRight()) {
			MPI_Isend(&(this->_workerMatrix[(_myNumberOfPlanes - 1) * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
			MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
			MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[_myNumberOfPlanes * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
			MPI_FLOAT, this->_mpiRank + 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
			MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));
		} else if (haveGhostCellsZLeft() && haveGhostCellsZRight()) {
			MPI_Isend(&(this->_workerMatrix[_numberOfCellsPerPlane]), _numberOfCellsPerPlane,
			MPI_FLOAT, this->_mpiRank - 1, GHOAST_PLANES_EXCHANGE_MSG_TAG,
			MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[0]));

			MPI_Irecv(&(this->_workerMatrix[0]), _numberOfCellsPerPlane, MPI_FLOAT, this->_mpiRank - 1,
			GHOAST_PLANES_EXCHANGE_MSG_TAG, MPI_COMM_WORLD, &(_exchangeGhostPlaneRequest[1]));

			MPI_Isend(&(this->_workerMatrix[(_myNumberOfPlanes) * _numberOfCellsPerPlane]), _numberOfCellsPerPlane,
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

template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsXLeft() {
	return false;
}
template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsX() {
	return false;
}
template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsYLeft() {
	return false;
}
template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsYRight() {
	return false;
}
template<typename T> bool SharedMatrix_ZPlanes<T>::haveGhostCellsZLeft() {
	return _myStartPlane > 0;
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

template<typename T> SharedMatrix_ZPlanes<T>::SharedMatrix_ZPlanes(int mpiRank, int mpiSize, int matX, int matY,
		int matZ) :
		SharedMatrix<T>(mpiRank, mpiSize, matX, matY, matZ), _numberOfCellsPerPlane(matX * matY) {

	// Do init Calculations
	int numberOfWorkers = this->_mpiSize - 1;
	_maxNumberOfPlanesPerWorker = (int) ceil((double) this->_matZ / (double) numberOfWorkers);
	_numberOfParticipantWorker = (this->_matZ < numberOfWorkers) ? this->_matZ : numberOfWorkers;
	_restNumberOfPlanesForLastWorker = this->_matZ - (_maxNumberOfPlanesPerWorker * (_numberOfParticipantWorker - 1));
	if (mpiRank == 0) {
		_myNumberOfPlanes = 0;
	} else if (_numberOfParticipantWorker == 1 && this->_mpiRank == 1) {
		_myNumberOfPlanes = _maxNumberOfPlanesPerWorker;

	} else {
		if (this->_mpiRank < _numberOfParticipantWorker) {
			_myNumberOfPlanes = _maxNumberOfPlanesPerWorker;
		} else if (this->_mpiRank == _numberOfParticipantWorker) {
			_myNumberOfPlanes = _restNumberOfPlanesForLastWorker;
		}
	}

	if (_myNumberOfPlanes > 0) {
		_myStartPlane = (this->_mpiRank - 1) * _maxNumberOfPlanesPerWorker;
		_myNumberOfGhostPlanes += haveGhostCellsZLeft() == true ? 1 : 0;
		_myNumberOfGhostPlanes += haveGhostCellsZRight() == true ? 1 : 0;
	}

	if (doIParticipate()) {
		this->_workerMatrix = new T[(_myNumberOfGhostPlanes + _myNumberOfPlanes) * _numberOfCellsPerPlane];
		_exchangeGhostPlaneRequest = new MPI_Request[_myNumberOfGhostPlanes * 2];
		_exchangeGhostPlaneStatus = new MPI_Status[_myNumberOfGhostPlanes * 2];
	}

	if (this->_mpiRank == MPI_MASTER) {
		this->_masterMatrix = new T[this->_matX * this->_matY * this->_matZ];
		_mergeMasterMatrixRequestsMASTER = new MPI_Request[_numberOfParticipantWorker];
		_mergeMasterMatrixStatusMASTER = new MPI_Status[_numberOfParticipantWorker];
		_distributeMasterMatrixRequestsMASTER = new MPI_Request[_numberOfParticipantWorker];
		_distributeMasterMatrixStatusMASTER = new MPI_Status[_numberOfParticipantWorker];
	}
	//std::cout << "Rank " << this->_mpiRank << " NumberOfPlanes " << _myNumberOfPlanes << std::endl;
	/*int sum;
	 MPI_Reduce(&_myNumberOfPlanes, &sum, 1, MPI_INT, MPI_SUM, 0,
	 MPI_COMM_WORLD);
	 MPI_Barrier(MPI_COMM_WORLD);
	 if (this->_mpiRank == 0) {
	 if (sum != matZ)
	 std::cout << "Error: " << sum << "!=" << matZ << std::endl;
	 }*/

}
template<typename T> SharedMatrix_ZPlanes<T>::~SharedMatrix_ZPlanes() {

	if (_mergeMasterMatrixRequestsMASTER != nullptr)
		delete[] _mergeMasterMatrixRequestsMASTER;

	if (_distributeMasterMatrixRequestsMASTER != nullptr)
		delete[] _distributeMasterMatrixRequestsMASTER;

	if (_distributeMasterMatrixStatusMASTER != nullptr)
		delete[] _distributeMasterMatrixStatusMASTER;

	if (_mergeMasterMatrixStatusMASTER != nullptr)
		delete[] _mergeMasterMatrixStatusMASTER;

	if (_exchangeGhostPlaneRequest != nullptr)
		delete[] _exchangeGhostPlaneRequest;

	if (_exchangeGhostPlaneStatus != nullptr)
		delete[] _exchangeGhostPlaneStatus;

}

#endif //SHAREDMATRIX_ZPLANES_HPP
