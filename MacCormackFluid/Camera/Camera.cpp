#include "../StdAfx.hpp"
#include "Camera.hpp"

Camera::Camera(GLFWwindow* window)
{
	this->window = window;

	glfwGetWindowSize(window, &windowWidth, &windowHeigt);
	//glfwSetCursor(window, GLUT_CURSOR_NONE);
	//glfwSetScrollCallback(window, *this.ScrollCallback);
}

 void Camera::ScrollCallback(GLFWwindow* window, double x, double y)
{

}

Camera::~Camera()
{
}

void Camera::computeMatricesFromInputs()
{
	// glfwGetTime is called only once, the first time this function is called
	static double lastTime = glfwGetTime();

	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// Get mouse position
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Reset mouse position for next frame
	glfwSetCursorPos(window, windowWidth / 2, windowHeigt / 2);

	// Compute new orientation
	horizontalAngle += mouseSpeed * float(windowWidth / 2 - xpos);
	verticalAngle += mouseSpeed * float(windowHeigt / 2 - ypos);

	// Direction : Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
		);

	// Right vector
	glm::vec3 right = glm::vec3(
		sin(horizontalAngle - 3.14f / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
		);

	// Up vector
	glm::vec3 up = glm::cross(right, direction);

	// Move forward
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
		position += direction * deltaTime * speed;
	}
	// Move backward
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
		position -= direction * deltaTime * speed;
	}
	// Strafe right
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
		position += right * deltaTime * speed;
	}
	// Strafe left
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
		position -= right * deltaTime * speed;
	}

	float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	ProjectionMatrix = glm::perspective(glm::radians(FoV), (float)windowWidth / (float)windowHeigt, 0.1f, 100.0f);
	// Camera matrix
	ViewMatrix = glm::lookAt(
		position,           // Camera is here
		position + direction, // and looks here : at the same position, plus "direction"
		up                  // Head is up (set to 0,-1,0 to look upside-down)
		);

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;
}
glm::mat4 Camera::getViewMatrix()
{
	return ViewMatrix;
}
glm::mat4 Camera::getProjectionMatrix()
{
	return ProjectionMatrix;
}