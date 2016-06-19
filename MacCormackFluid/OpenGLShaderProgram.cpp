#include "StdAfx.hpp"
#include "OpenGLShaderProgram.hpp"

#include "ResourceGuard.hpp"

#include <fstream>

class OpenGLShaderPrivate {
public:
    OpenGLShaderPrivate(OpenGLShader::Type type)
        : compiled(false)
        , shaderType(type)
        , shaderGuard(nullptr) {}

    ~OpenGLShaderPrivate() {
        destroy();
    }

    bool create();
    void destroy();
    bool compile(OpenGLShader* q);
    bool isValid() const;

    std::string log;
    bool compiled;
    OpenGLShader::Type shaderType;
    ResourceGuard<GLuint>* shaderGuard;
};

namespace {
    void freeShaderFunc(GLuint id) {
        glDeleteShader(id);
    }
}

bool OpenGLShaderPrivate::create() {
    if (isValid()) {
        return true;
    }

    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        destroy();
        shaderGuard = new ResourceGuard<GLuint>(shader, freeShaderFunc);
        return isValid();
    } else {
        std::cout << "Could not create shader" << std::endl;
        return false;
    }
}

void OpenGLShaderPrivate::destroy() {
    if (shaderGuard) {
        delete shaderGuard;
        shaderGuard = nullptr;
    }
}

bool OpenGLShaderPrivate::compile(OpenGLShader* q) {
    if (!isValid()) {
        std::cout << "Shader not created" << std::endl;
        return false;
    }

    GLuint shader = shaderGuard->get();
    glCompileShader(shader);
    GLint value = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &value);
    compiled = (value != 0);

    if (!compiled) {
        const char* types[] = {
            "Vertex",
            "Fragment",
            "Geometry",
            "Tessellation Control",
            "Tessellation Evaluation",
            "Compute",
            ""
        };

        const char* type = types[6];
        switch (shaderType) {
        case OpenGLShader::Vertex:
            type = types[0]; break;
        case OpenGLShader::Fragment:
            type = types[1]; break;
        case OpenGLShader::Geometry:
            type = types[2]; break;
        case OpenGLShader::TessellationControl:
            type = types[3]; break;
        case OpenGLShader::TessellationEvaluation:
            type = types[4]; break;
        case OpenGLShader::Compute:
            type = types[5]; break;
        }

        GLint infoLogLength = 0;
        char* logBuffer = nullptr;

        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        if (infoLogLength > 1) {
            GLint tmp;
            logBuffer = new char[infoLogLength];
            glGetShaderInfoLog(shader, infoLogLength, &tmp, logBuffer);
        }

        if (logBuffer) {
            log = std::string(logBuffer);
        }

        std::cout << "OpenGLShader::compile(" << type << "): " << log;
        std::cout << "*** Problematic " << type << " shader source code ***";
        std::cout << q->getSourceCode();
        std::cout << "***";

        delete[] logBuffer;
    }

    return compiled;
}

bool OpenGLShaderPrivate::isValid() const {
    return (shaderGuard) && (shaderGuard->get() != 0);
}

OpenGLShader::OpenGLShader(OpenGLShader::Type type)
    : dPtr(new OpenGLShaderPrivate(type)) {

    D(OpenGLShader);
    d->create();
}

OpenGLShader::~OpenGLShader() {
    destroy();
}

OpenGLShader::Type OpenGLShader::getShaderType() const {
    D(const OpenGLShader);
    return d->shaderType;
}

bool OpenGLShader::compileSourceCode(const char* source, int length) {
    if (isCreated() && source) {
        D(OpenGLShader);
        glShaderSource(d->shaderGuard->get(), 1, &source, &length);
        return d->compile(this);
    } else {
        std::cout << "Shader not created";
        return false;
    }
}

bool OpenGLShader::compileSourceCode(const std::string& source) {
    return compileSourceCode(source.c_str(), source.length());
}

bool OpenGLShader::compileSourceFile(const std::string& fileName) {
    std::ifstream stream(fileName, std::ios::in);
    std::string source;

    if (stream.is_open()) {
        std::string line;
        while (getline(stream, line)) {
            source.append(line + "\n");
        }
    } else {
        std::cout << "Unable to load shader file:" << fileName;
        return false;
    }

    return compileSourceCode(source);
}

void OpenGLShader::destroy() {
    if (dPtr) {
        delete dPtr;
        dPtr = nullptr;
    }
}

std::string OpenGLShader::getSourceCode() const {
    if (isCreated()) {
        D(const OpenGLShader);
        GLuint shader = d->shaderGuard->get();
        GLint sourceCodeLength = 0;
        glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &sourceCodeLength);
        if (sourceCodeLength <= 0) {
            return std::string();
        }
        GLint length = 0;
        char* source = new char[sourceCodeLength];
        glGetShaderSource(shader, sourceCodeLength, &length, source);
        std::string sourceCode(source);
        delete[] source;
        return sourceCode;
    } else {
        std::cout << "Shader not created";
        return std::string();
    }
}

std::string OpenGLShader::getLog() const {
    D(const OpenGLShader);
    return d->log;
}

bool OpenGLShader::isCreated() const {
    D(const OpenGLShader);
    return d->isValid();
}

bool OpenGLShader::isCompiled() const {
    D(const OpenGLShader);
    return d->compiled;
}

GLuint OpenGLShader::getId() const {
    D(const OpenGLShader);
    return (d->shaderGuard) ? (d->shaderGuard->get()) : (0);
}

class OpenGLShaderProgramPrivate {
public:
    OpenGLShaderProgramPrivate()
        : linked(false)
        , inited(false)
        , removingShaders(false)
        , programGuard(nullptr) {}

    ~OpenGLShaderProgramPrivate() {
        destroy();
    }

    bool create();
    void destroy();
    bool hasShader(OpenGLShader::Type type) const;
    bool isValid() const;

    bool linked;
    bool inited;
    bool removingShaders;
    ResourceGuard<GLuint>* programGuard;

    std::string log;
    std::list<OpenGLShader *> shaders;
    std::list<OpenGLShader *> anonShaders;
};

namespace {
    void freeProgramFunc(GLuint id) {
        glDeleteProgram(id);
    }
}

bool OpenGLShaderProgramPrivate::create() {
    if (isValid() || inited) {
        return true;
    }

    inited = true;
    GLuint program = glCreateProgram();
    if (program) {
        if (programGuard) {
            delete programGuard;
            programGuard = nullptr;
        }
        programGuard = new ResourceGuard<GLuint>(program, freeProgramFunc);
        return isValid();
    } else {
        std::cout << "Could not create shader program";
        return false;
    }
}

void OpenGLShaderProgramPrivate::destroy() {
    if (programGuard) {
        delete programGuard;
        programGuard = nullptr;
    }
}

bool OpenGLShaderProgramPrivate::hasShader(OpenGLShader::Type type) const {
    for (const auto& shader : shaders) {
        if (shader->getShaderType() == type) {
            return true;
        }
    }
    return false;
}

bool OpenGLShaderProgramPrivate::isValid() const {
    return (programGuard) && (programGuard->get() != 0);
}

OpenGLShaderProgram::OpenGLShaderProgram()
    : dPtr(new OpenGLShaderProgramPrivate) {}

OpenGLShaderProgram::~OpenGLShaderProgram() {
    destroy();
}

bool OpenGLShaderProgram::addShader(OpenGLShader* shader) {
    if (create() && shader) {
        if (isCreated() && shader->isCreated()) {
            D(OpenGLShaderProgram);
            glAttachShader(d->programGuard->get(), shader->getId());
            d->linked = false;
            d->shaders.push_back(shader);
            return true;
        } else {
            std::cout << "Shader not created";
            return false;
        }
    } else {
        std::cout << "Shader program not created";
        return false;
    }
}

void OpenGLShaderProgram::removeShader(OpenGLShader* shader) {
    if (isCreated() && shader) {
        if (shader->isCreated()) {
            D(OpenGLShaderProgram);
            glDetachShader(d->programGuard->get(), shader->getId());
            d->linked = false;
            d->shaders.remove(shader);
            d->anonShaders.remove(shader);
        } else {
            std::cout << "Shader not created";
        }
    } else {
        std::cout << "Shader program not created";
    }
}

const std::list<OpenGLShader *>& OpenGLShaderProgram::getShaders() const {
    D(const OpenGLShaderProgram);
    return d->shaders;
}

bool OpenGLShaderProgram::addShaderFromSourceCode(OpenGLShader::Type type, const char* source) {
    if (create() && source) {
        D(OpenGLShaderProgram);
        OpenGLShader* shader = new OpenGLShader(type);
        if (shader->compileSourceCode(source)) {
            d->anonShaders.push_back(shader);
            return addShader(shader);
        } else {
            d->log = shader->getLog();
            delete shader;
            return false;
        }
    } else {
        std::cout << "Shader program not created";
        return false;
    }
}

bool OpenGLShaderProgram::addShaderFromSourceCode(OpenGLShader::Type type, const std::string& source) {
    return addShaderFromSourceCode(type, source.c_str());
}

bool OpenGLShaderProgram::addShaderFromSourceFile(OpenGLShader::Type type, const std::string& fileName) {
    if (create()) {
        D(OpenGLShaderProgram);
        OpenGLShader* shader = new OpenGLShader(type);
        if (shader->compileSourceFile(fileName)) {
            d->anonShaders.push_back(shader);
            return addShader(shader);
        } else {
            d->log = shader->getLog();
            delete shader;
            return false;
        }
    } else {
        std::cout << "Shader program not created";
        return false;
    }
}

void OpenGLShaderProgram::removeAllShaders() {
    if (isCreated()) {
        D(OpenGLShaderProgram);
        d->removingShaders = true;
        for (const auto& shader : d->shaders) {
            if (shader && shader->isCreated()) {
                glDetachShader(d->programGuard->get(), shader->getId());
            }
        }
        for (const auto& shader : d->anonShaders) {
            delete shader;
        }
        d->shaders.clear();
        d->anonShaders.clear();
        d->linked = false;
        d->removingShaders = false;
    } else {
        std::cout << "Shader program not created";
    }
}

bool OpenGLShaderProgram::create() {
    D(OpenGLShaderProgram);
    return d->create();
}

bool OpenGLShaderProgram::isCreated() const {
    D(const OpenGLShaderProgram);
    return d->isValid();
}

void OpenGLShaderProgram::destroy() {
    removeAllShaders();
    if (dPtr) {
        delete dPtr;
        dPtr = nullptr;
    }
}

bool OpenGLShaderProgram::link() {
    if (isCreated()) {
        D(OpenGLShaderProgram);
        GLuint program = d->programGuard->get();
        GLint value = 0;
        if (d->shaders.empty()) {
            value = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &value);
            d->linked = (value != 0);
            if (d->linked) {
                return true;
            }
        }
        glLinkProgram(program);
        value = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &value);
        d->linked = (value != 0);
        value = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &value);
        d->log = std::string();
        if (value > 1) {
            char* logBuffer = new char[value];
            GLint length = 0;
            glGetProgramInfoLog(program, value, &length, logBuffer);
            d->log = std::string(logBuffer);
            if (!d->linked) {
                std::cout << d->log;
            }
            delete[] logBuffer;
        }
        return d->linked;
    } else {
        std::cout << "Shader program not created";
        return false;
    }
}

bool OpenGLShaderProgram::isLinked() const {
    D(const OpenGLShaderProgram);
    return d->linked;
}

const std::string& OpenGLShaderProgram::getLog() const {
    D(const OpenGLShaderProgram);
    return d->log;
}

bool OpenGLShaderProgram::bind() {
    if (isCreated()) {
        D(OpenGLShaderProgram);
        if (d->linked) {
            glUseProgram(d->programGuard->get());
            return true;
        } else {
            std::cout << "Shader program must be linked first";
            return false;
        }
    } else {
        std::cout << "Shader program not created";
        return false;
    }
}

void OpenGLShaderProgram::release() {
    if (!isCreated()) {
        std::cout << "Shader program not created";
    }
    glUseProgram(0);
}

GLuint OpenGLShaderProgram::getId() const {
    D(const OpenGLShaderProgram);
    return d->programGuard ? d->programGuard->get() : 0;
}

int OpenGLShaderProgram::getMaxGeometryOutputVertices() const {
    GLint geometryOutputVertices = 0;
    glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &geometryOutputVertices);
    return geometryOutputVertices;
}

void OpenGLShaderProgram::setPatchVertexCount(int count) {
    glPatchParameteri(GL_PATCH_VERTICES, count);
}

int OpenGLShaderProgram::getPatchVertexCount() const {
    GLint patchVertices = 0;
    glGetIntegerv(GL_PATCH_VERTICES, &patchVertices);
    return patchVertices;
}

void OpenGLShaderProgram::setDefaultOuterTessellationLevels(const std::vector<float>& levels) {
    std::vector<float> tessLevels = levels;

    const int argCount = 4;
    if (tessLevels.size() < argCount) {
        tessLevels.reserve(argCount);
        for (size_t i = tessLevels.size(); i < argCount; ++i) {
            tessLevels.push_back(1.0f);
        }
    }

    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, tessLevels.data());
}

std::vector<float> OpenGLShaderProgram::getDefaultOuterTessellationLevels() const {
    std::vector<float> tessLevels(4, 1.0f);
    glGetFloatv(GL_PATCH_DEFAULT_OUTER_LEVEL, tessLevels.data());
    return tessLevels;
}

void OpenGLShaderProgram::setDefaultInnerTessellationLevels(const std::vector<float>& levels) {
    std::vector<float> tessLevels = levels;

    const int argCount = 2;
    if (tessLevels.size() < argCount) {
        tessLevels.reserve(argCount);
        for (size_t i = tessLevels.size(); i < argCount; ++i) {
            tessLevels.push_back(1.0f);
        }
    }

    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, tessLevels.data());
}

std::vector<float> OpenGLShaderProgram::getDefaultInnerTessellationLevels() const {
    std::vector<float> tessLevels(2, 1.0f);
    glGetFloatv(GL_PATCH_DEFAULT_INNER_LEVEL, tessLevels.data());
    return tessLevels;
}

void OpenGLShaderProgram::bindAttributeLocation(const char* name, int location) {
    if (isCreated() && name) {
        D(OpenGLShaderProgram);
        glBindAttribLocation(d->programGuard->get(), location, name);
        d->linked = false;
    } else {
        std::cout << "OpenGLShaderProgram::bindAttributeLocation(" << name
                     << "): Shader program not created";
    }
}

void OpenGLShaderProgram::bindAttributeLocation(const std::string& name, int location) {
    bindAttributeLocation(name.c_str(), location);
}

int OpenGLShaderProgram::getAttributeLocation(const char* name) const {
    if (isCreated() && isLinked() && name) {
        D(const OpenGLShaderProgram);
        return glGetAttribLocation(d->programGuard->get(), name);
    } else {
        std::cout << "OpenGLShaderProgram::getAttributeLocation(" << name
                     << "): Shader program is not linked";
        return -1;
    }
}

int OpenGLShaderProgram::getAttributeLocation(const std::string& name) const {
    return getAttributeLocation(name.c_str());
}

void OpenGLShaderProgram::setAttributeValue(int location, GLfloat value) {
    if (location >= 0) {
        glVertexAttrib1fv(location, &value);
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, GLfloat x, GLfloat y) {
    if (location >= 0) {
        GLfloat values[2] = {x, y};
        glVertexAttrib2fv(location, values);
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z) {
    if (location >= 0) {
        GLfloat values[3] = {x, y, z};
        glVertexAttrib3fv(location, values);
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    if (location >= 0) {
        GLfloat values[4] = {x, y, z, w};
        glVertexAttrib4fv(location, values);
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec2& value) {
    if (location >= 0) {
        glVertexAttrib2fv(location, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec3& value) {
    if (location >= 0) {
        glVertexAttrib3fv(location, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec4& value) {
    if (location >= 0) {
        glVertexAttrib4fv(location, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setAttributeValue(const char* name, GLfloat value) {
    setAttributeValue(getAttributeLocation(name), value);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, GLfloat x, GLfloat y) {
    setAttributeValue(getAttributeLocation(name), x, y);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z) {
    setAttributeValue(getAttributeLocation(name), x, y, z);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    setAttributeValue(getAttributeLocation(name), x, y, z, w);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, const glm::vec2& value) {
    setAttributeValue(getAttributeLocation(name), value);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, const glm::vec3& value) {
    setAttributeValue(getAttributeLocation(name), value);
}

void OpenGLShaderProgram::setAttributeValue(const char* name, const glm::vec4& value) {
    setAttributeValue(getAttributeLocation(name), value);
}

void OpenGLShaderProgram::setAttributeArray(int location, const GLfloat* values, int tupleSize, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, tupleSize, GL_FLOAT, GL_FALSE, stride, values);
    }
}

void OpenGLShaderProgram::setAttributeArray(int location, const glm::vec2* values, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, stride, values);
    }
}

void OpenGLShaderProgram::setAttributeArray(int location, const glm::vec3* values, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, stride, values);
    }
}

void OpenGLShaderProgram::setAttributeArray(int location, const glm::vec4* values, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, stride, values);
    }
}

void OpenGLShaderProgram::setAttributeArray(int location, GLenum type, const void* values, int tupleSize, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, tupleSize, type, GL_FALSE, stride, values);
    }
}

void OpenGLShaderProgram::setAttributeArray(const char* name, const GLfloat* values, int tupleSize, int stride) {
    setAttributeArray(getAttributeLocation(name), values, tupleSize, stride);
}

void OpenGLShaderProgram::setAttributeArray(const char* name, const glm::vec2* values, int stride) {
    setAttributeArray(getAttributeLocation(name), values, stride);
}

void OpenGLShaderProgram::setAttributeArray(const char* name, const glm::vec3* values, int stride) {
    setAttributeArray(getAttributeLocation(name), values, stride);
}

void OpenGLShaderProgram::setAttributeArray(const char* name, const glm::vec4* values, int stride) {
    setAttributeArray(getAttributeLocation(name), values, stride);
}

void OpenGLShaderProgram::setAttributeArray(const char* name, GLenum type, const void* values, int tupleSize, int stride) {
    setAttributeArray(getAttributeLocation(name), type, values, tupleSize, stride);
}

void OpenGLShaderProgram::setAttributeBuffer(int location, GLenum type, int offset, int tupleSize, int stride) {
    if (location >= 0) {
        glVertexAttribPointer(location, tupleSize, type, GL_FALSE, stride, reinterpret_cast<const void *>(offset));
    }
}

void OpenGLShaderProgram::setAttributeBuffer(const char* name, GLenum type, int offset, int tupleSize, int stride) {
    setAttributeBuffer(getAttributeLocation(name), type, offset, tupleSize, stride);
}

void OpenGLShaderProgram::enableAttributeArray(int location) {
    if (location >= 0) {
        glEnableVertexAttribArray(location);
    }
}

void OpenGLShaderProgram::enableAttributeArray(const char* name) {
    enableAttributeArray(getAttributeLocation(name));
}

void OpenGLShaderProgram::disableAttributeArray(int location) {
    if (location >= 0) {
        glDisableVertexAttribArray(location);
    }
}

void OpenGLShaderProgram::disableAttributeArray(const char* name) {
    disableAttributeArray(getAttributeLocation(name));
}

int OpenGLShaderProgram::getUniformLocation(const char* name) const {
    if (isCreated() && isLinked() && name) {
        D(const OpenGLShaderProgram);
        return glGetUniformLocation(d->programGuard->get(), name);
    } else {
        std::cout << "OpenGLShaderProgram::getUniformLocation(" << name
                     << "): Shader program is not linked";
        return -1;
    }
}

int OpenGLShaderProgram::getUniformLocation(const std::string& name) const {
    return getUniformLocation(name.c_str());
}

void OpenGLShaderProgram::setUniformValue(int location, GLfloat value) {
    if (location >= 0) {
        glUniform1fv(location, 1, &value);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, GLint value) {
    if (location >= 0) {
        glUniform1i(location, value);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, GLuint value) {
    if (location >= 0) {
        glUniform1ui(location, value);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, GLfloat x, GLfloat y) {
    if (location >= 0) {
        GLfloat values[2] = {x, y};
        glUniform2fv(location, 1, values);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z) {
    if (location >= 0) {
        GLfloat values[3] = {x, y, z};
        glUniform3fv(location, 1, values);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    if (location >= 0) {
        GLfloat values[4] = {x, y, z, w};
        glUniform4fv(location, 1, values);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::vec2& value) {
    if (location >= 0) {
        glUniform2fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::vec3& value) {
    if (location >= 0) {
        glUniform3fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::vec4& value) {
    if (location >= 0) {
        glUniform4fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat2x2& value) {
    if (location >= 0) {
        glUniformMatrix2fv(location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat2x3& value) {
    if (location >= 0) {
        glUniform3fv(location, 2, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat2x4& value) {
    if (location >= 0) {
        glUniform4fv(location, 2, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat3x2& value) {
    if (location >= 0) {
        glUniform2fv(location, 3, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat3x3& value) {
    if (location >= 0) {
        glUniformMatrix3fv(location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat3x4& value) {
    if (location >= 0) {
        glUniform4fv(location, 3, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat4x2& value) {
    if (location >= 0) {
        glUniform2fv(location, 4, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat4x3& value) {
    if (location >= 0) {
        glUniform3fv(location, 4, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const glm::mat4x4& value) {
    if (location >= 0) {
        glUniformMatrix4fv(location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&value));
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[2][2]) {
    if (location >= 0) {
        glUniformMatrix2fv(location, 1, GL_FALSE, value[0]);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[3][3]) {
    if (location >= 0) {
        glUniformMatrix3fv(location, 1, GL_FALSE, value[0]);
    }
}

void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[4][4]) {
    if (location >= 0) {
        glUniformMatrix4fv(location, 1, GL_FALSE, value[0]);
    }
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLfloat value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLint value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLuint value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLfloat x, GLfloat y) {
    setUniformValue(getUniformLocation(name), x, y);
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z) {
    setUniformValue(getUniformLocation(name), x, y, z);
}

void OpenGLShaderProgram::setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    setUniformValue(getUniformLocation(name), x, y, z, w);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::vec2& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::vec3& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::vec4& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat2x2& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat2x3& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat2x4& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat3x2& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat3x3& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat3x4& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat4x2& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat4x3& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const glm::mat4x4& value) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const GLfloat value[2][2]) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const GLfloat value[3][3]) {
    setUniformValue(getUniformLocation(name), value);
}

void OpenGLShaderProgram::setUniformValue(const char* name, const GLfloat value[4][4]) {
    setUniformValue(getUniformLocation(name), value);
}