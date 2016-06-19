#ifndef CONSTANT_MEMORY_HPP
#define CONSTANT_MEMORY_HPP

template <class Symbol>
inline void moveHostToSymbol(Symbol& symbol, const Symbol& source) {
    void* target = nullptr;
    cudaGetSymbolAddress(&target, symbol);
    ERRORCHECK_CUDA();
    moveHostToDevice(target, &source, sizeof(source));
}

#endif