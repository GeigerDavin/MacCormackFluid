#ifndef OPENGL_SHADER_PROGRAM_HPP
#define OPENGL_SHADER_PROGRAM_HPP

namespace OpenGL {

class OpenGLShaderPrivate;

class OpenGLShader {
    DISABLE_COPY(OpenGLShader)

public:
    enum Type {
        Vertex                 = GL_VERTEX_SHADER,
        Fragment               = GL_FRAGMENT_SHADER,
        Geometry               = GL_GEOMETRY_SHADER,
        TessellationControl    = GL_TESS_CONTROL_SHADER,
        TessellationEvaluation = GL_TESS_EVALUATION_SHADER,
        Compute                = GL_COMPUTE_SHADER
    };

public:
    explicit OpenGLShader(OpenGLShader::Type type);
    ~OpenGLShader();

public:
    OpenGLShader::Type getShaderType() const;

    bool compileSourceCode(const char* source, int length);
    bool compileSourceCode(const std::string& source);
    bool compileSourceFile(const std::string& fileName);

    void destroy();

    std::string getSourceCode() const;
    std::string getLog() const;

    bool isCreated() const;
    bool isCompiled() const;
    GLuint getId() const;

private:
    DECLARE_PRIVATE(OpenGLShader)

    OpenGLShaderPrivate* dPtr;
};

class OpenGLShaderProgramPrivate;

class OpenGLShaderProgram {
    DISABLE_COPY(OpenGLShaderProgram)

public:
    explicit OpenGLShaderProgram();
    ~OpenGLShaderProgram();

public:
    bool addShader(OpenGLShader* shader);
    void removeShader(OpenGLShader* shader);
    const std::list<OpenGLShader *>& getShaders() const;

    bool addShaderFromSourceCode(OpenGLShader::Type type, const char* source);
    bool addShaderFromSourceCode(OpenGLShader::Type type, const std::string& source);
    bool addShaderFromSourceFile(OpenGLShader::Type type, const std::string& fileName);

    void removeAllShaders();

public:
    bool create();
    bool isCreated() const;

    void destroy();

    bool link();
    bool isLinked() const;
    const std::string& getLog() const;

    bool bind();
    void release();

    GLuint getId() const;

public:
    int getMaxGeometryOutputVertices() const;

    void setPatchVertexCount(int count);
    int getPatchVertexCount() const;

    void setDefaultOuterTessellationLevels(const std::vector<float>& levels);
    std::vector<float> getDefaultOuterTessellationLevels() const;

    void setDefaultInnerTessellationLevels(const std::vector<float>& levels);
    std::vector<float> getDefaultInnerTessellationLevels() const;

public:
    void bindAttributeLocation(const char* name, int location);
    void bindAttributeLocation(const std::string& name, int location);

    int getAttributeLocation(const char* name) const;
    int getAttributeLocation(const std::string& name) const;

    void setAttributeValue(int location, GLfloat value);
    void setAttributeValue(int location, GLfloat x, GLfloat y);
    void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z);
    void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    void setAttributeValue(int location, const glm::vec2& value);
    void setAttributeValue(int location, const glm::vec3& value);
    void setAttributeValue(int location, const glm::vec4& value);

    void setAttributeValue(const char* name, GLfloat value);
    void setAttributeValue(const char* name, GLfloat x, GLfloat y);
    void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z);
    void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    void setAttributeValue(const char* name, const glm::vec2& value);
    void setAttributeValue(const char* name, const glm::vec3& value);
    void setAttributeValue(const char* name, const glm::vec4& value);

    void setAttributeArray(int location, const GLfloat* values, int tupleSize, int stride = 0);
    void setAttributeArray(int location, const glm::vec2* values, int stride = 0);
    void setAttributeArray(int location, const glm::vec3* values, int stride = 0);
    void setAttributeArray(int location, const glm::vec4* values, int stride = 0);
    void setAttributeArray(int location, GLenum type, const void* values, int tupleSize, int stride = 0);

    void setAttributeArray(const char* name, const GLfloat* values, int tupleSize, int stride = 0);
    void setAttributeArray(const char* name, const glm::vec2* values, int stride = 0);
    void setAttributeArray(const char* name, const glm::vec3* values, int stride = 0);
    void setAttributeArray(const char* name, const glm::vec4* values, int stride = 0);
    void setAttributeArray(const char* name, GLenum type, const void* values, int tupleSize, int stride = 0);

    void setAttributeBuffer(int location, GLenum type, int offset, int tupleSize, int stride = 0);
    void setAttributeBuffer(const char* name, GLenum type, int offset, int tupleSize, int stride = 0);

    void enableAttributeArray(int location);
    void enableAttributeArray(const char* name);
    void disableAttributeArray(int location);
    void disableAttributeArray(const char* name);

    int getUniformLocation(const char* name) const;
    int getUniformLocation(const std::string& name) const;

    void setUniformValue(int location, GLfloat value);
    void setUniformValue(int location, GLint value);
    void setUniformValue(int location, GLuint value);
    void setUniformValue(int location, GLfloat x, GLfloat y);
    void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z);
    void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    void setUniformValue(int location, const glm::vec2& value);
    void setUniformValue(int location, const glm::vec3& value);
    void setUniformValue(int location, const glm::vec4& value);
    void setUniformValue(int location, const glm::mat2x2& value);
    void setUniformValue(int location, const glm::mat2x3& value);
    void setUniformValue(int location, const glm::mat2x4& value);
    void setUniformValue(int location, const glm::mat3x2& value);
    void setUniformValue(int location, const glm::mat3x3& value);
    void setUniformValue(int location, const glm::mat3x4& value);
    void setUniformValue(int location, const glm::mat4x2& value);
    void setUniformValue(int location, const glm::mat4x3& value);
    void setUniformValue(int location, const glm::mat4x4& value);
    void setUniformValue(int location, const GLfloat value[2][2]);
    void setUniformValue(int location, const GLfloat value[3][3]);
    void setUniformValue(int location, const GLfloat value[4][4]);

    void setUniformValue(const char* name, GLfloat value);
    void setUniformValue(const char* name, GLint value);
    void setUniformValue(const char* name, GLuint value);
    void setUniformValue(const char* name, GLfloat x, GLfloat y);
    void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z);
    void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    void setUniformValue(const char* name, const glm::vec2& value);
    void setUniformValue(const char* name, const glm::vec3& value);
    void setUniformValue(const char* name, const glm::vec4& value);
    void setUniformValue(const char* name, const glm::mat2x2& value);
    void setUniformValue(const char* name, const glm::mat2x3& value);
    void setUniformValue(const char* name, const glm::mat2x4& value);
    void setUniformValue(const char* name, const glm::mat3x2& value);
    void setUniformValue(const char* name, const glm::mat3x3& value);
    void setUniformValue(const char* name, const glm::mat3x4& value);
    void setUniformValue(const char* name, const glm::mat4x2& value);
    void setUniformValue(const char* name, const glm::mat4x3& value);
    void setUniformValue(const char* name, const glm::mat4x4& value);
    void setUniformValue(const char* name, const GLfloat value[2][2]);
    void setUniformValue(const char* name, const GLfloat value[3][3]);
    void setUniformValue(const char* name, const GLfloat value[4][4]);

private:
    DECLARE_PRIVATE(OpenGLShaderProgram)

    OpenGLShaderProgramPrivate* dPtr;
};

} // namespace OpenGL

#endif