#include "gl/MY_GLSL.h"
#include "gl/MY_MATH.h"

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "CCSManager.h"

#include "GL/glh_glut.h"
#include <fstream>

//TODO: timer
//#include "timer.hpp"
//double g_timing_start = 0;
#include <time.h>

bool b[256];
int win_w = 512, win_h = 512;

using namespace glh;
glut_simple_mouse_interactor object;
void CaptureScreen(int, int);

float DISP_SCALE = 0.001f;
char *dataPath;
char *animPath = NULL;
int stFrame = 0;
int pFrame = stFrame;

bool use_adf_force = true;

bool wireframe = false;

GLuint depth_FBO = 0;
GLuint depth_texture = 0;

GLuint shadow_program = 0;
GLuint phong_program = 0;

float	zoom = 30;
float	swing_angle = 0;
float	elevate_angle = 10;
float	center[3] = { 0, -0.2, 0 };

#define NO_MOTION			0
#define ZOOM_MOTION			1
#define ROTATE_MOTION		2
#define TRANSLATE_MOTION	3

int motionMode = NO_MOTION;
int mousex = 0;
int mousey = 0;

// for sprintf
#pragma warning(disable: 4996)

extern void offsetMesh();
extern void old_stitch();

extern void initModel(int, char **, char *, int);
extern void quitModel();
extern bool dynamicModel(char *, bool, int, char *);
extern void dumpModel();
extern void loadModel();
extern void checkModel();
extern void checkCCD();
extern void checkSelf();
extern bool checkSelfIJ(int, int);
extern void exportMeshes(char *, int);

static int level = 1;

//float lightpos[4] = { 13, 10.2, 3.2, 0 };
float lightpos[3] = { -2, 2, 4 };


float tick()
{
	return GetTickCount()*0.001f;
}

// check for OpenGL errors
void checkGLError()
{
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR) {
		char msg[512];
		sprintf(msg, "error - %s\n", (char *)gluErrorString(error));
		printf(msg);
	}
}

static char fpsBuffer[512];

void CalculateFrameRate()
{
#ifdef TODO_CODE
	static float framesPerSecond = 0.0f;       // This will store our fps
	static float lastTime = 0.0f;       // This will hold the time from the last frame
	static bool first = true;
	static Timer keeper;
	//static System.Windows.Forms.Timer keeper;

	if (first) {
		keeper.tick();
		first = false;
	}
	float currentTime = keeper.tock2();

	++framesPerSecond;
	float delta = currentTime - lastTime;
	if (currentTime - lastTime > 1.0f)
	{
		lastTime = currentTime;
		//if (SHOW_FPS == 1) fprintf(stderr, "\nCurrent Frames Per Second: %d\n\n", (int)framesPerSecond);
		sprintf(fpsBuffer, "Cloth Simulator (%s)------- (FPS: %d)", use_adf_force ? "with ADF" : "without ADF", int(ceil(framesPerSecond)));
		framesPerSecond = 0;
	}
#endif
	static float framesPerSecond = 0.0f;
	static float lastTime = 0.0f;
	static bool first = true;
	if (first) {
		lastTime = GetTickCount()*0.001f;
		first = false;
	}
	float currentTime = GetTickCount()*0.001f;

	++framesPerSecond;
	float delta = currentTime - lastTime;
	if (delta > 1.0f) {
		lastTime = currentTime;
		//if (SHOW_FPS == 1) fprintf(stderr, "\nCurrent Frames Per Second: %d\n\n", (int)framesPerSecond);
		sprintf(fpsBuffer, "Cloth Simulator (%s,%s)------- (FPS: %d, frame %d)",
			use_adf_force ? "with ADF" : "without ADF",
			b['o'] ? "output" : "no output",
			int(ceil(framesPerSecond)), pFrame);
		framesPerSecond = 0;
	}

}

void initSetting()
{
	b['9'] = false;
	b['d'] = false;
}

void initOpengl()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// initialize OpenGL lighting
	GLfloat lightPos[] = { 10.0, 10.0, 10.0, 0.0 };
	GLfloat lightAmb[4] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat lightDiff[4] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat lightSpec[4] = { 1.0, 1.0, 1.0, 1.0 };

	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec);

	//glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL_EXT, GL_SEPARATE_SPECULAR_COLOR_EXT);
	GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
}

void updateFPS()
{
	CalculateFrameRate();
	glutSetWindowTitle(fpsBuffer);
}

void begin_window_coords()
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, win_w, 0.0, win_h, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void end_window_coords()
{
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

void drawGround()
{
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	glBegin(GL_QUADS);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(20, 0, 20);
	glVertex3f(-20, 0, 20);
	glVertex3f(-20, 0, -20);
	glVertex3f(20, 0, -20);
	glEnd();

	glDisable(GL_COLOR_MATERIAL);
}

extern void drawEdges(bool, bool);
extern void drawVFs(int);
extern void drawDebugVF(int);
extern void drawMeshes(bool w);
extern void draw_handles();
extern void visualize_cones();

static void drawModel(bool, bool, bool, bool, int)
{
	/*glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1, 1);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);*/

	drawMeshes(wireframe);

	/*glColor4d(0, 0, 0, 0.2);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glEnable(GL_LIGHTING);*/

	glDisable(GL_LIGHTING);
	if (true) {
			glLineWidth(3.0f);
			visualize_cones();
			draw_handles();
	}
	glEnable(GL_LIGHTING);
}


void draw()
{
	glPushMatrix();
	glRotatef(-90, 1, 0, 0);

	drawModel(!b['t'], !b['p'], !b['m'], b['e'], level);

	glPopMatrix();

	if (b['g'])
		drawGround();
}

static bool ret = false;

void Create_Shadow_Map(char* filename = 0)
{
#ifdef XXXXXXX
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	glViewport(0, 0, 1024, 1024); // Render on the whole framebuffer, complete from the lower left corner to the upper right

	// glEnable(GL_CULL_FACE);
	// glCullFace(GL_BACK); // Cull back-facing triangles -> draw only front-facing triangles
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2, 2, -2, 2, 0, 20);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(lightpos[0], lightpos[1], lightpos[2], 0, 0, 0, 0, 1, 0);
	//Use fixed program
	glUseProgram(0);

	glPushMatrix();
	glRotated(elevate_angle, 1, 0, 0);
	glRotated(swing_angle, 0, 1, 0);
	glTranslatef(center[0], center[1], -center[2]);
	draw();
	glPopMatrix();

	//Also we need to set up the projection matrix for shadow texture	
	// This is matrix transform every coordinate x,y,z
	// Moving from unit cube [-1,1] to [0,1]  
	float bias[16] = { 0.5, 0.0, 0.0, 0.0,
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
		0.5, 0.5, 0.5, 1.0 };

	// Grab modelview and transformation matrices
	float	modelView[16];
	float	projection[16];
	float	biased_MVP[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glGetFloatv(GL_PROJECTION_MATRIX, projection);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLoadMatrixf(bias);
	// concatating all matrice into one.
	glMultMatrixf(projection);
	glMultMatrixf(modelView);

	glGetFloatv(GL_MODELVIEW_MATRIX, biased_MVP);

	glUseProgram(shadow_program);
	GLuint m = glGetUniformLocation(shadow_program, "biased_MVP"); // get the location of the biased_MVP matrix
	glUniformMatrix4fv(m, 1, GL_FALSE, biased_MVP);
#endif
}

void display()
{
	Create_Shadow_Map();

	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, win_w, win_h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(4, (double)win_w / (double)win_h, 1, 100);
	glMatrixMode(GL_MODELVIEW);
	glShadeModel(GL_SMOOTH);

	glLoadIdentity();
	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0);

	//glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glShadeModel(GL_SMOOTH);

	glUseProgram(shadow_program);
	GLuint uniloc = glGetUniformLocation(shadow_program, "shadow_texture");
	glUniform1i(uniloc, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depth_texture);

	uniloc = glGetUniformLocation(shadow_program, "light_position");
	glUniform3fv(uniloc, 1, lightpos);

	glBegin(GL_POLYGON);
	glVertex3f(-10, -10, -1);
	glVertex3f(10, -10, -1);
	glVertex3f(10, 10, -1);
	glVertex3f(-10, 10, -1);
	glEnd();

	/*if (!b['b']) {
	// gradient background
	begin_window_coords();
	glBegin(GL_QUADS);
	glColor3f(0.2, 0.4, 0.8);
	glVertex2f(0.0, 0.0);
	glVertex2f(win_w, 0.0);
	glColor3f(0.05, 0.1, 0.2);
	glVertex2f(win_w, win_h);
	glVertex2f(0, win_h);
	glEnd();
	end_window_coords();
	}

	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();
	object.apply_transform();*/

	// draw scene
	if (b['w'])
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	/*glEnable(GL_LIGHTING);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);*/

	glRotated(elevate_angle, 1, 0, 0);
	glRotated(swing_angle, 0, 1, 0);
	glTranslatef(center[0], center[1], -center[2]);
	glUseProgram(phong_program);
	uniloc = glGetUniformLocation(phong_program, "light_position");
	glUniform3fv(uniloc, 1, lightpos);

	GLuint c0 = glGetAttribLocation(phong_program, "position");
	GLuint c1 = glGetAttribLocation(phong_program, "normal");
	glEnableVertexAttribArray(c0);
	glEnableVertexAttribArray(c1);

	draw();
	//glutSolidTeapot(1.0);

	glutSwapBuffers();
	updateFPS();
	//checkGLError();

	if (b['x'] && ret)   {
		CaptureScreen(512, 512);
	}
}

void idle()
{
	if (b[' '])
		object.trackball.increment_rotation();

	if (b['d'])
	{
		ret = dynamicModel(dataPath, b['o'], pFrame++, animPath);
	}

	glutPostRedisplay();
}

void key1()
{
	dynamicModel(dataPath, b['o'], pFrame++, animPath);
	glutPostRedisplay();
}

void key2()
{
	checkModel();
}

bool loadVtx(char *cfile, char *ofile, bool orig);

void key6()
{
	use_adf_force = !use_adf_force;
}

void key3()
{
}

void key9()
{
#ifdef TODO_CODE
	//dynamicModel(dataPath, false, true);
	gStitcher.begin();
#endif

	glutPostRedisplay();
}

void quit()
{
	quitModel();
	exit(0);
}

void printLight()
{
	//printf("Light: %f, %f, %f, %f\n", lightpos[0], lightpos[1], lightpos[2], lightpos[3]);
	printf("Light: %f, %f, %f\n", lightpos[0], lightpos[1], lightpos[2]);
}

void updateLight()
{
	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
}

void endCapture()
{
}

void key(unsigned char k, int x, int y)
{
	b[k] = !b[k];

	switch (k) {
	case 't':
		exportMeshes(dataPath, pFrame);
		break;
	case 27:
	case 'q':
		quit();
		break;

	case 'x':
	{
				if (b['x'])
					printf("Starting screen capturing.\n");
				else
					printf("Ending screen capturing.\n");

				break;
	}

		// adjust light source
	case 'L':
		lightpos[0] += 0.2f;
		updateLight();
		break;

	case 'J':
		lightpos[0] -= 0.2f;
		updateLight();
		break;

	case 'I':
		lightpos[1] += 0.2f;
		updateLight();
		break;

	case 'K':
		lightpos[1] -= 0.2f;
		updateLight();
		break;

	case 'O':
		lightpos[2] += 0.2f;
		updateLight();
		break;

	case 'U':
		lightpos[2] -= 0.2f;
		updateLight();
		break;

	case 'r':
		initModel(0, NULL, dataPath, stFrame);
		pFrame = stFrame;
		break;

	case 'w':
		wireframe = !wireframe;
		break;

	case '1':
		key1();
		break;

	case '2':
		key2();
		break;

	case '=':
		level++;
		break;

	case '-':
		level--;
		break;

	case '3':
		key3();
		key2();
		break;

	case '6':
		key6();
		break;

	case '9':
		key9();
		break;

	case 's':
		key1();
		break;

#ifdef TODO_CODE
	case '7':
		offsetMesh();
		break;

	case '8':
		old_stitch();
		break;
#endif
	}

	object.keyboard(k, x, y);
	glutPostRedisplay();
}

void resize(int w, int h)
{
	if (h == 0) h = 1;

	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.1, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	object.reshape(w, h);

	win_w = w; win_h = h;
}

//==========================================================
#ifdef TODO_CODE
extern double matModelView[], matProjection[];
extern int select_pbd_cloth(float *, float *);
extern int select_cloth(float *pp, float *qq);

float vv[3];
float hitted[3];
float ray1[3], ray2[3];
float p[3], q[3];
int		select_v = -1;
float	target[3] = { 0, 0, 0 };




#define ADD(a, b, c)	{c[0]=a[0]+b[0]; c[1]=a[1]+b[1]; c[2]=a[2]+b[2];}
#define SUB(a, b, c)	{c[0]=a[0]-b[0]; c[1]=a[1]-b[1]; c[2]=a[2]-b[2];}
#define DOT(a, b)		(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define CROSS(a, b, r)	{r[0]=a[1]*b[2]-a[2]*b[1]; r[1]=a[2]*b[0]-a[0]*b[2]; r[2]=a[0]*b[1]-a[1]*b[0];}
#define	MIN(a,b)		((a)<(b)?(a):(b))
#define	MAX(a,b)		((a)>(b)?(a):(b))
#define SQR(a)			((a)*(a))
#define CLAMP(a, l, h)  (((a)>(h))?(h):(((a)<(l))?(l):(a)))
#define SIGN(a)			((a)<0?-1:1)
#define SWAP(X, Y)      {temp={X}; X=(Y); Y=(temp);}

#define Distance2(x, y) ((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]))

template <class T> FORCEINLINE
T Magnitude(T *x)
{
	return sqrtf(DOT(x, x));
}


template <class T> FORCEINLINE
T Dot(T *v0, T *v1)
{
	return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

template <class T> FORCEINLINE
void Cross(T* a, T* b, T* r)
{
	r[0] = a[1] * b[2] - a[2] * b[1];
	r[1] = a[2] * b[0] - a[0] * b[2];
	r[2] = a[0] * b[1] - a[1] * b[0];
}

template <class T> FORCEINLINE
T Normalize(T *x)
{
	T m = Magnitude(x);
	if (m<1e-14f)	return m;//{printf("ERROR: vector cannot be normalized.\n"); return m;}
	T inv_m = 1 / m;
	x[0] *= inv_m;
	x[1] *= inv_m;
	x[2] *= inv_m;
	return m;
}


template <class T>
void Matrix_Vector_Product_4(T *A, T *x, T *r)
{
	r[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2] + A[3] * x[3];
	r[1] = A[4] * x[0] + A[5] * x[1] + A[6] * x[2] + A[7] * x[3];
	r[2] = A[8] * x[0] + A[9] * x[1] + A[10] * x[2] + A[11] * x[3];
	r[3] = A[12] * x[0] + A[13] * x[1] + A[14] * x[2] + A[15] * x[3];
}


template <class TYPE>
static void Get_Selection_Ray(int mouse_x, int mouse_y, TYPE* p, TYPE* q)
{
	int viewport[4];

	// get matrix and viewport:
	glGetIntegerv(GL_VIEWPORT, viewport);

	// window pos of mouse, Y is inverted on Windows
	double winX = (double)mouse_x;
	double winY = win_h - (double)mouse_y;
	float winZ;
	glReadPixels(winX, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
	printf("Z = %lf\n", winZ);

	double start[3], end[3];

	// get point on the 'near' plane (third param is set to 0.0)
	gluUnProject(winX, winY, 0, matModelView, matProjection,
		viewport, start, start + 1, start + 2);

	// get point on the 'far' plane (third param is set to 1.0)
	gluUnProject(winX, winY, 1.0, matModelView, matProjection,
		viewport, end, end + 1, end + 2);

	// now you can create a ray from m_start to m_end
	p[0] = start[0], p[1] = start[1], p[2] = start[2];
	q[0] = end[0], q[1] = end[1], q[2] = end[2];
}
#endif

void mouse(int button, int state, int x, int y)
{
#ifdef TODO_CODE
	if (state == GLUT_DOWN) {
		Get_Selection_Ray(x, y, p, q);
		select_v = select_cloth(p, q);

		ray1[0] = p[0], ray1[1] = p[1], ray1[2] = p[2];
		ray2[0] = q[0], ray2[1] = q[1], ray2[2] = q[2];

		// Set up the motion target
		if (select_v != -1)
		{
			printf("hitted!\n");
			float pos[3];
			pos[0] = vv[0], pos[1] = vv[1], pos[2] = vv[2];
			hitted[0] = pos[0], hitted[1] = pos[1], hitted[2] = pos[2];
			target[0] = pos[0], target[1] = pos[1], target[2] = pos[2];

			/*
			double dir[3];
			dir[0] = q[0] - p[0];
			dir[1] = q[1] - p[1];
			dir[2] = q[2] - p[2];
			Normalize(dir);
			double diff[3];

			diff[0] = pos[0] - p[0];
			diff[1] = pos[1] - p[1];
			diff[2] = pos[2] - p[2];
			double dist = DOT(diff, dir);
			CLAMP(dist, -0.001, 0.001);

			target[0] = p[0] + dist*dir[0];
			target[1] = p[1] + dist*dir[1];
			target[2] = p[2] + dist*dir[2];
			*/

		}
	}
	else
	if (state == GLUT_UP) {
		select_v = -1;
	}
#endif

	object.mouse(button, state, x, y);
}

void newmouse(int button, int state, int x, int y) {
	if (state == GLUT_UP) motionMode = NO_MOTION;
	if (state == GLUT_DOWN) {
		int modif = glutGetModifiers();
		if (modif & GLUT_ACTIVE_SHIFT)		motionMode = TRANSLATE_MOTION;
		else if (modif & GLUT_ACTIVE_CTRL)	motionMode = ZOOM_MOTION;
		else								motionMode = ROTATE_MOTION;
		mousex = x;
		mousey = y;
	}
}

void motion(int x, int y)
{
#ifdef TODO_CODE
	if (select_v != -1) {
		float	p[3], q[3];
		Get_Selection_Ray(x, y, p, q);
		double dir[3];
		dir[0] = q[0] - p[0];
		dir[1] = q[1] - p[1];
		dir[2] = q[2] - p[2];
		Normalize(dir);
		double diff[3];

		float pos[3];
		//get_pbd_pos(select_v, pos);
		pos[0] = vv[0], pos[1] = vv[1], pos[2] = vv[2];

		diff[0] = pos[0] - p[0];
		diff[1] = pos[1] - p[1];
		diff[2] = pos[2] - p[2];
		double dist = DOT(diff, dir);
		CLAMP(dist, -0.001, 0.001);

		target[0] = p[0] + dist*dir[0];
		target[1] = p[1] + dist*dir[1];
		target[2] = p[2] + dist*dir[2];
		glutPostRedisplay();
		return;
	}
#endif

	object.motion(x, y);
}

void newmotion(int x, int y) {
	if (motionMode != NO_MOTION)
	{
		if (motionMode == ROTATE_MOTION)
		{
			swing_angle += (double)(x - mousex) * 360 / (double)win_w;
			elevate_angle += (double)(y - mousey) * 180 / (double)win_h;
			if (elevate_angle> 90)	elevate_angle = 90;
			else if (elevate_angle<-90)	elevate_angle = -90;
		}
		if (motionMode == ZOOM_MOTION)	zoom += 0.05 * (y - mousey);
		if (motionMode == TRANSLATE_MOTION)
		{
			center[0] -= 0.01*(mousex - x);
			center[2] += 0.01*(mousey - y);
		}
		mousex = x;
		mousey = y;
		glutPostRedisplay();
	}
}

void main_menu(int i)
{
	key((unsigned char)i, 0, 0);
}

void initMenu()
{
	glutCreateMenu(main_menu);
	glutAddMenuEntry("Toggle animation [d]", 'd');
	glutAddMenuEntry("Toggle output [o]", 'o');
	glutAddMenuEntry("Toggle wireframe [w]", 'w');
	glutAddMenuEntry("========================", '=');
	glutAddMenuEntry("Quit/q [esc]", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void init_GLSL()
{
#ifdef XXXXXXXXXXXXXXXXXXX
	//Init GLEW
	GLenum err = glewInit();
	if (err != GLEW_OK)  printf(" Error initializing GLEW! \n");
	else printf("Initializing GLEW succeeded!\n");

	//Init depth texture and FBO
	glGenFramebuffers(1, &depth_FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	// Depth texture. Slower than a depth buffer, but you can sample it later in your shader
	glGenTextures(1, &depth_texture);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);
	glDrawBuffer(GL_NONE);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("Init_Shadow_Map failed.\n");

	//Load shader program
	char shadow[512], phong[512];
	sprintf(shadow, ".\\shadow", dataPath);
	sprintf(phong, ".\\phong", dataPath);
	shadow_program = Setup_GLSL(shadow);
	phong_program = Setup_GLSL(phong);
#endif
}

//#define NO_UI
#ifdef NO_UI

int main(int argc, char **argv)
{
	if (argc < 2)
		printf("usage: %s data_path [start_frame] \n", argv[0]);

	dataPath = argv[1];

	if (argc == 3) {
		sscanf(argv[2], "%d", &stFrame);
	}

	//#define MUTE_TOTALLY
#ifdef MUTE_TOTALLY
	std::streambuf* cout_sbuf = std::cout.rdbuf();
	std::ofstream fout("/dev/null");
	std::cout.rdbuf(fout.rdbuf());

	FILE *stream;
	if ((stream = freopen("file.txt", "w", stdout)) == NULL)
		exit(-1);
#endif

	initModel(argc, argv, dataPath, stFrame);
	dynamicModel(dataPath, true, stFrame++);

	g_timing_start = omp_get_wtime();
	while (1)
		dynamicModel(dataPath, true, stFrame++);

	quit();
	return 0;
}

#else

int main(int argc, char **argv)
{
	if (argc < 2)
		printf("usage: %s data_path [start_frame] [anim_path]\n", argv[0]);

	dataPath = argv[1];

	if (argc >= 3) {
		sscanf(argv[2], "%d", &stFrame);
	}
	if (argc >= 4) {
		animPath = strdup(argv[3]);
	}

	//#define MUTE_TOTALLY
#ifdef MUTE_TOTALLY
	std::streambuf* cout_sbuf = std::cout.rdbuf();
	std::ofstream fout("/dev/null");
	std::cout.rdbuf(fout.rdbuf());

	FILE *stream;
	if ((stream = freopen("file.txt", "w", stdout)) == NULL)
		exit(-1);
#endif

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow("Cloth Simulator");

	//initOpengl();
	initModel(argc, argv, dataPath, stFrame);

	//object.configure_buttons(1);
	//object.dolly.dolly[2] = -3;
	//object.trackball.incr = rotationf(vec3f(1, 1, 0), 0.05);

	glutDisplayFunc(display);
	glutMouseFunc(newmouse);
	glutMotionFunc(newmotion);
	glutIdleFunc(idle);
	glutKeyboardFunc(key);
	glutReshapeFunc(resize);

	initMenu();

	initSetting();
	init_GLSL();
	glutMainLoop();

	quit();
	return 0;
}
#endif

void CaptureScreen(int Width, int Height)
{
	static int captures = 0;
	char filename[512];

	sprintf(filename, "Data/%04d.bmp", captures);
	captures++;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;

	char *image = new char[Width*Height * 3];
	FILE *file = fopen(filename, "wb");

	if (image != NULL)
	{
		if (file != NULL)
		{
			glReadPixels(0, 0, Width, Height, GL_BGR_EXT, GL_UNSIGNED_BYTE, image);

			memset(&bf, 0, sizeof(bf));
			memset(&bi, 0, sizeof(bi));

			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf)+sizeof(bi)+Width*Height * 3;
			bf.bfOffBits = sizeof(bf)+sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = Width;
			bi.biHeight = Height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = Width*Height * 3;

			fwrite(&bf, sizeof(bf), 1, file);
			fwrite(&bi, sizeof(bi), 1, file);
			fwrite(image, sizeof(unsigned char), Height*Width * 3, file);

			fclose(file);
		}
		delete[] image;
	}
}

void drawPt(float x, float y, float z)
{
	glColor3f(1, 0, 0);
	glBegin(GL_POINTS);
	glVertex3f(x, y, z);
	glEnd();

}

/*void initRedMat(int side)
{
GLfloat matAmb[4] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat matDiff[4] = { 1.0, 0.1, 0.2, 1.0 };
GLfloat matSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
glMaterialfv(side, GL_AMBIENT, matAmb);
glMaterialfv(side, GL_DIFFUSE, matDiff);
glMaterialfv(side, GL_SPECULAR, matSpec);
glMaterialf(side, GL_SHININESS, 600.0);
}

void initBlueMat(int side)
{
GLfloat matAmb[4] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat matDiff[4] = { 0.0, 1.0, 1.0, 1.0 };
GLfloat matSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
glMaterialfv(side, GL_AMBIENT, matAmb);
glMaterialfv(side, GL_DIFFUSE, matDiff);
glMaterialfv(side, GL_SPECULAR, matSpec);
glMaterialf(side, GL_SHININESS, 60.0);
}

void initYellowMat(int side)
{
GLfloat matAmb[4] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat matDiff[4] = { 1.0, 1.0, 0.0, 1.0 };
GLfloat matSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
glMaterialfv(side, GL_AMBIENT, matAmb);
glMaterialfv(side, GL_DIFFUSE, matDiff);
glMaterialfv(side, GL_SPECULAR, matSpec);
glMaterialf(side, GL_SHININESS, 60.0);
}

void initGrayMat(int side)
{
GLfloat matAmb[4] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat matDiff[4] = { 0.5, 0.5, 0.5, 1.0 };
GLfloat matSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
glMaterialfv(side, GL_AMBIENT, matAmb);
glMaterialfv(side, GL_DIFFUSE, matDiff);
glMaterialfv(side, GL_SPECULAR, matSpec);
glMaterialf(side, GL_SHININESS, 60.0);
}

void initSkinMat(int side)
{
GLfloat matAmb[4] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat matDiff[4] = { 0.98, 0.92, 0.84, 1.0 };
GLfloat matSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
glMaterialfv(side, GL_AMBIENT, matAmb);
glMaterialfv(side, GL_DIFFUSE, matDiff);
glMaterialfv(side, GL_SPECULAR, matSpec);
glMaterialf(side, GL_SHININESS, 60.0);
}*/

void drawMeshInfo(float *vtxs, float *nrms, int *tris, int numTri, int type, bool w)
{
	float color[3];
	if (type == 0) {
		color[0] = 0.98f;
		color[1] = 0.92f;
		color[2] = 0.84f;
	}
	else {
		if (type % 2 == 0) {//yellow
			color[0] = 1.0f;
			color[1] = 0.9f;
			color[2] = 0.0f;
		}
		else if (type % 3 == 0) {//red
			color[0] = 1.0f;
			color[1] = 0.3f;
			color[2] = 0.4f;
		}
		else {//blue
			color[0] = 0.0f;
			color[1] = 0.8f;
			color[2] = 1.0f;
		}
	}
	
	glUseProgram(phong_program);
	GLuint uniloc = glGetUniformLocation(phong_program, "objColor");
	glUniform3fv(uniloc, 1, color);

	//glPolygonOffset(5.f, 5.f);

	if (w) {//wireframe
		for (int i = 0; i < numTri * 3; i += 3) {
			glBegin(GL_LINE_LOOP);
			glVertex3f(vtxs[tris[i] * 3], vtxs[tris[i] * 3 + 1], vtxs[tris[i] * 3 + 2]);
			glVertex3f(vtxs[tris[i + 1] * 3], vtxs[tris[i + 1] * 3 + 1], vtxs[tris[i + 1] * 3 + 2]);
			glVertex3f(vtxs[tris[i + 2] * 3], vtxs[tris[i + 2] * 3 + 1], vtxs[tris[i + 2] * 3 + 2]);
			glEnd();
		}
	}
	else {
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glVertexPointer(3, GL_FLOAT, sizeof(float)* 3, vtxs);
		glNormalPointer(GL_FLOAT, sizeof(float)* 3, nrms);
		glDrawElements(GL_TRIANGLES, numTri * 3, GL_UNSIGNED_INT, tris);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
	}

	//glDisable(GL_POLYGON_OFFSET_FILL);

}
