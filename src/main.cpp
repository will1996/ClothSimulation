
#include <fstream>

#include "real.hpp"
#include<math.h>
#include "timer.hpp"
double g_timing_start = 0;

bool b[256];
int win_w = 512, win_h = 512;

void CaptureScreen(int, int);

float DISP_SCALE = 0.001f;
char *dataPath;
int stFrame = 0;

// for sprintf
#pragma warning(disable: 4996)

extern void key_s();
extern void key_S();
extern void initModel(int, char **, char *, int);
extern void quitModel();
extern void drawModel(bool, bool, bool, bool, int);
extern void updateModel();
extern bool dynamicModel(char *, bool, bool);
extern void dumpModel();
extern void loadModel();
extern void checkModel();
extern void checkCCD();
extern void checkSelf();
extern bool checkSelfIJ(int, int);

static int level = 1;

float lightpos[4] = {13, 10.2, 3.2, 0};


static char fpsBuffer[512];

void CalculateFrameRate()
{
	static float framesPerSecond = 0.0f;       // This will store our fps
	static float lastTime = 0.0f;       // This will hold the time from the last frame
	static bool first = true;
	static Timer keeper;

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
		sprintf(fpsBuffer, "Cloth Simulator ------ (FPS: %d)", int(ceil(framesPerSecond)));
		framesPerSecond = 0;
	}
}

void initSetting()
{
	b['9'] = false;
	b['d'] = false;
}
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
	dynamicModel(dataPath, false, false);

	g_timing_start = omp_get_wtime();
	while(1)
		dynamicModel(dataPath, false, false);

	return 0;
}


extern void init_resume(char *path, int st);
extern void init_cuda(int argc, char **argv);

void initModel(int argc, char **argv, char *path, int st)
{
	init_cuda(argc, argv);
	init_resume(path, st);

}

void quitModel()
{

}
