#ifndef VECTOR_HPP
#define VECTOR_HPP

template <class T>
class Vector {
public:
    Vector(unsigned int size);
    ~Vector();

public:
    inline void sendData();
    inline void recvData();
    inline void setChanged(bool gpu);

public:
    inline size_t getSize() const {
        return size;
    }
    inline T* getHost() {
        return elementsHost;
    }
    inline const T* getHost() const {
        return elementsHost;
    }
    inline T* getDevice() {
        return elementsDevice;
    }
    inline const T* getDevice() const {
        return elementsDevice;
    }

private:
    int location;
    size_t size;
    unsigned T* elementsHost;
    unsigned T* elementsDevice;
};

#endif