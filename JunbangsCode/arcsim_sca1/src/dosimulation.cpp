
#include "dosimulation.hpp"

#include "conf.hpp"
#include "io.hpp"
#include "misc.hpp"
#include "separateobs.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include "util.hpp"
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <cstdio>
#include <fstream>
#include <json/json.h>
#include <ctime>
#include <cstdlib>
using namespace std;

bool prepared = false;
Timer timer, prefix;
MPI_Request r0, r1, r2;
MPI_Status status;
bool checked = false, shouldSendM = true;
int M = 0, M0 = 0;
int myrank, largeStep;
vector<Mesh> tmp_mesh;

void sim_step(int level = 0, int rest_level = 0, Timer *timer=NULL, int largeStep=1, int reload = 0);
void save (Simulation &sim, int frame, int level = 0, int step = 0);
void copy_file (const string &input, const string &output);
extern void parse_handles (vector<Handle*>&, const Json::Value&,
                     const vector<Cloth>&, const vector<Motion>&);
extern void parse_obstacles (vector<Obstacle>&, const Json::Value&,
                      const vector<Motion>&, bool finer = false);
void update_obstacles (Simulation &sim, bool update_positions=true);
void refine_mesh(Simulation &sim);


void checkDx(Simulation &sim, const char *s)
{
        Mesh &mesh = *sim.cloth_meshes[0];
        Vec3 dv = mesh.nodes[0]->dv;
        Vec3 v = mesh.nodes[0]->v;
    printf("rank %d step %d %s ",myrank,sim.step,s);
    cout << "dv "<< dv[0]<<" "<<dv[1]<<" "<<dv[2]<<" v "<< v[0]<<" "<<v[1]<<" "<<v[2];
    cout << endl;
}

void initialize(Simulation &sim, int start_step, int end_step, int cur_level, int rest_level, 
    int pstart, int pend, int source, string outprefix)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    printf("simulating at processor %d, step interval [%d, %d], cur_level=%d, rest_level=%d, \
        rest processor=[%d, %d]",
      myrank, start_step, end_step, cur_level, rest_level, pstart, pend);
    cout << endl;
    if (!prepared)
    {
        prepared=true;
        prepare(sim);
    }
    //resume from last level
    sim.frame = start_step / sim.frame_steps;
    sim.time = start_step * sim.step_time;
    sim.step = start_step;
    sim.end_frame = end_step;
    sim.end_step = end_step;
    load_objs(sim.cloth_meshes, stringf("%s/%d_%04d_%02d",outprefix.c_str(),cur_level-1,
        sim.frame,sim.step%sim.frame_steps));
    sim.coarseMesh_real.resize(sim.cloth_meshes.size());
    sim.coarseMesh.resize(sim.cloth_meshes.size());
    for (int i = 0; i < sim.cloth_meshes.size(); ++i)
        sim.coarseMesh[i] = &sim.coarseMesh_real[i];
    //cout << "coarsesize = " << sim.coarseMesh.size() << endl;
    for (int i = 0 ; i < sim.cloths.size(); ++i)
    {
        double tmp = sim.cloths[i].remeshing.tmp_min;
        for (int k = 0; k < rest_level; ++k)
            tmp *= _coarseFineRatio;
        sim.cloths[i].remeshing.size_min = tmp;
        cout << "current size=" << sim.cloths[i].remeshing.size_min << endl;
    }
    // Reparse the handles
    {
        Json::Value json;
        Json::Reader reader;
        string configFilename = stringf("%s/conf.json", outprefix.c_str());
        ifstream file(configFilename.c_str());
        bool parsingSuccessful = reader.parse(file, json);
        if(!parsingSuccessful) {
            fprintf(stderr, "Error reading file: %s\n", configFilename.c_str());
            fprintf(stderr, "%s", reader.getFormatedErrorMessages().c_str());
            abort();
        }
        file.close();
        sim.handles.clear();
        parse_handles(sim.handles, json["handles"], sim.cloths, sim.motions);
        if (rest_level == 0)
        {
            sim.obstacles.clear();
            parse_obstacles(sim.obstacles, json["obstacles"], sim.motions, true);
        }
    }
    prepare(sim); // re-prepare the new cloth meshes
    update_obstacles(sim,true);
    for (int c = 0; c < sim.obstacle_meshes.size(); c++) {
       compute_ws_data(*sim.obstacle_meshes[c]);
       update_x0(*sim.obstacle_meshes[c]);
    }
    separate_obstacles(sim.obstacle_meshes, sim.cloth_meshes);
    //collision_step(sim);
    //cout << myrank << " before remeshing" << endl;
    remeshing_step(sim);
    if (!sim.enabled[Simulation::Remeshing]){
    if (cur_level > 1)
        for (int i = 0; i < sim.l1; ++i)
            refine_mesh(sim);
    else
	for (int i = 0; i < sim.l0; ++i)
		refine_mesh(sim);
    }
    //relax_initial_state(sim);
    //cout << myrank << " finish init" << endl;
    //save_obj(sim.obstacles[0].get_mesh(), stringf("%s/newobs%d.obj", outprefix.c_str(), myrank));
}

int determinePartition(int rest_level, int pstart, int pend)
{
	int ans = 1;
    if (rest_level != 0)
    {
        for (; ; ++ans)
        {
            int tmp = pend-pstart+1;
            for (int i = 0; i < rest_level; ++i)
                tmp /= ans;
            if (tmp == 0)
            {
                --ans;
                break;
            }
        }
    }
    return ans;
}

void do_static_step(Simulation &sim, int start_step, string outprefix)
{
    if (start_step == 0) return;
//return;
    //int static_state_step = sqrt(2*_coarseFineRatio/(_coarseFineRatio-1)/sim.step_time);
    //int static_state_step = 4*sqrt(_coarseFineRatio*sim.cloths[0].remeshing.size_min/10)/sim.step_time;
    //int static_state_step = double(3.1415926535)/sqrt(sim.step_time);
    double alpha = 0.0;//1-pow(1-0.2, 50*sim.step_time);
    int static_state_step = 10;//log(0.1)/log(1-alpha);
cout << alpha << " " << static_state_step << endl;
    int static_rest_step = static_state_step;
if (static_rest_step > sim.step) static_rest_step = sim.step;
    int largeStep = 1;
    for (int i = 0; i < static_rest_step; ++i)
    {
        {
            //cout << "back_step"<<i<<" before!" << myrank << endl;
            //save_objs(sim.cloth_meshes, stringf("%s/tmp%d_%04d_%03dbefore", outprefix.c_str(), 2, sim.frame, i));
            back_step(sim, 1);
            //save_objs(sim.cloth_meshes, stringf("%s/tmp%d_%04d_%03dafter", outprefix.c_str(), 2, sim.frame, i));
            //cout << "back_step"<<i<<" success!" << myrank << endl;
            sim.add_pos = true;
            advance_step(sim, 0, alpha, 1);
            sim.add_pos = false;
            //cout << "advance_step"<<i<<" success!" << myrank << endl;
        }
        if (i % largeStep == largeStep-1)
        {
            //cout << "remeshing!" << i/largeStep << endl;
            remeshing_step(sim, false);
            //save_objs(sim.cloth_meshes, stringf("%s/tmp%d_%04d_%03dsim", outprefix.c_str(), 2, sim.frame, i));
        }
    }
    if (sim.step % sim.frame_steps == 0)
        save(sim, sim.frame, 2, 0);
}

void do_simulation(Simulation &sim, int start_step, int end_step, int cur_level, int rest_level, 
    int pstart, int pend, int source, string outprefix)
{
    sim.lev = cur_level;
    if (start_step == 2) start_step = 0;
    sim.finer = false;
    if (source != -1)
        MPI_Irecv(&end_step, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &r0);
    else if (rest_level != 0)
        MPI_Irecv(&M0, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &r1);
    int tmpStep = sim.frame_steps/4;
    largeStep = 1;
    if (cur_level == 1)
        largeStep = tmpStep;
    initialize(sim, start_step, end_step, cur_level, rest_level, pstart, pend, source, outprefix);
    //determine partition number
    int ans = determinePartition(rest_level, pstart, pend);
    //simulate and recursive one
    int next_step = start_step+tmpStep, cur_part = 0;
    timer.tick();
    double sumtime = 0, sumnum = 0;
    double decay = 0.9;
    save(sim, sim.frame, cur_level, sim.step%sim.frame_steps);
    int lastsend = 0;
    vector<double> pretime;
    double fn0 = 0, prefn0 = 0, K = 1, baseK = 1;
    int totstep = 0, M1 = 0;
    do_static_step(sim, start_step, outprefix);
    //checkDx(sim, "before simulation ");
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (cur_level == 2 && rank != 0)
            sim.finer = true;
    }
    while (1)
    {
        //early stop
        if (checked && sim.frame == end_step / sim.frame_steps && rest_level == 0)
        {
            timer.tock();
            cout << "early stop! final level proc "<< myrank << " take time " << timer.last << endl;
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
        //partition
        if (sim.step == next_step && rest_level != 0)
        {
            prefix.total = 0;
            fn0 = prefn0 = 0;
            K *= 0.8; baseK = 1;
            M=M1=0;
            pretime.clear();
            //save(sim, sim.frame, cur_level, sim.step%sim.frame_steps);
            int param[10];
            next_step = end_step;
            start_step = sim.step;
            param[0] = sim.step, param[1] = end_step;
            param[2] = cur_level+1, param[3] = rest_level-1;
            int plen = (pend-pstart+1)/(ans-cur_part)+((pend-pstart+1)%(ans-cur_part)!=0);
            param[4] = pend-plen+1, param[5] = pend;
            lastsend = param[4];
            pend -= plen;
            ++cur_part;
            if (cur_part == ans)
            {
                timer.tock();
                cout << "level " << cur_level << " processor " << pstart << 
                " take time " << timer.last << endl;
                do_simulation(sim, sim.step, end_step, cur_level+1, rest_level-1, param[4], 
                	param[5],-1,outprefix);
                return;
            }
            else
            {
                /*cout << "sending to " << param[4] << " ";
                for (int i = 0; i < 6; ++i)
                    cout << param[i] << " ";
                cout << endl;*/
                MPI_Send(param, 6, MPI_INT, param[4], 0, MPI_COMM_WORLD);
            }
        }
        //simulate
        if (source != -1 && !checked)
        {
            int flag;
            MPI_Test(&r0, &flag, &status);
            if (flag)
            {
                checked = true;
                --end_step;
                //cout << "succesfully change end_step of " << myrank << " to " << end_step << endl;
                sim.end_step = end_step;
                shouldSendM = false;
            }
        }
        prefix.tick();
        if (cur_level==2 && sim.step % tmpStep == 0)
            sim_step(cur_level, rest_level, &timer, largeStep,sim.step + tmpStep);
        else
            sim_step(cur_level, rest_level, &timer, largeStep,0);
        if (cur_level==1)
            save(sim, sim.frame, cur_level, sim.step%sim.frame_steps);
        prefix.tock();
        ++totstep;
        pretime.push_back(prefix.total);
        sumtime = sumtime * decay + prefix.last;
        sumnum = sumnum * decay + 1;
        //calculate timing and determine partition
        //WARNING: ONLY WORKS ON TWO LEVELS!!!
        while (1)
        {
            if (source != -1)
            {
                if (!shouldSendM) break;
                MPI_Isend(&totstep, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &r1);
            }
            else
            {
                if (rest_level == 0) break;
                while (1)
                {
                    int flag;
                    MPI_Test(&r1, &flag, &status);
                    if (flag)
                    {
                        //cout << "received " << M0 << endl;
                        if (status.MPI_SOURCE == lastsend) M=M0;
                        MPI_Irecv(&M0, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &r1);
                    }
                    else
                        break;
                }
                if (M != M1)
                {
                    prefn0 = fn0;
                    M1=M;
                }
                        double avg = sumtime/sumnum;
                        fn0 = pretime.back();
                        int tot = end_step-start_step, n0 = sim.step-start_step;
                        avg=fn0/n0;
                        double tmp = fn0/(M<=0?avg:pretime[M/largeStep]);
                        //cout << "tmp="<<tmp<<" fn0=" << fn0 <<" M="<<M<<" timing="<<(M<=0?avg:pretime[M/largeStep])<<endl;
                        if (K < tmp) K = tmp;
//K=100000;
                        int b = ans-cur_part+1;
                        double q = 1-1.0/K;
                        double A = (1-q)/(1-pow(q,b-1))*pow(q,b-2)*(end_step-sim.step);
                        //cout << "tot " << tot << " n0 " << n0 << " b " << b << endl;
                        if ((K>1&&(K-1)*fn0+avg*n0 > avg*(tot+(K-1)*A)) || (n0+1)*b > tot+5*b*sim.frame_steps || tot-n0<(b+1)*largeStep || (n0+1)*b>tot*1.5)
                        {
                            printf("cur %d avg %lf fn0 %lf K %lf b %d q %lf A %lf tot %d n0 %d", 
                                sim.step, avg, fn0, K, b, q, A, tot, n0);
                            cout << endl;
                            MPI_Isend(&sim.step, 1, MPI_INT, lastsend, 0, MPI_COMM_WORLD, &r2);
                            next_step = sim.step;
                            M=0;
                        }
            }
            break;
        }
    }
}
