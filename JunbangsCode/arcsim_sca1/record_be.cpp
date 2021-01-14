#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <map>
using namespace std;

double ans;
const double PI = 3.1415926535;
class point
{
public:
	double x,y,z;
	point(double x0=0,double y0=0,double z0=0):x(x0),y(y0),z(z0){}
};
class edge
{
public:
	int p,q;
	edge(int p0=0,int q0=0):p(p0),q(q0){if(p>q)swap(p,q);}
};
vector<point> node;
point operator-(point a, point b){return point(a.x-b.x,a.y-b.y,a.z-b.z);}
point operator/(point a,double b){return point(a.x/b,a.y/b,a.z/b);}
double operator*(point a, point b){return a.x*b.x+a.y*b.y+a.z*b.z;}
point cross(point a, point b)
{
	return point(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
class face
{
public:
	int p,q,r;
	face(int p0=0,int q0=0,int r0=0):p(p0),q(q0),r(r0){}
	point getarea(){return cross(node[q]-node[p],node[r]-node[p]);}
};
bool operator<(edge a, edge b){if(a.p!=b.p)return a.p<b.p;return a.q<b.q;}
ostream &operator<<(ostream &out, point a)
{
	out << "(" << a.x << ","<<a.y<<","<<a.z<<")";
	return out;
}
ostream &operator<<(ostream &out, face a)
{
	out << "[" << a.p << a.q << a.r << "]";
	return out;
}
vector<face> faces;
map<edge, vector<int> > mesh;

bool consist(point e, face f)
{
	point tmp = node[f.q]-node[f.p]-e;
	if (tmp*tmp<1e-4) return true;
	tmp = node[f.r]-node[f.q]-e;
	if (tmp*tmp<1e-4) return true;
	tmp = node[f.p]-node[f.r]-e;
	if (tmp*tmp<1e-4) return true;
	return false;
}

double calcurv(point e, face f0, face f1)
{
	//curv=theta*length/area
	if (consist(e, f0)) swap(f0,f1);
	point a0 = f0.getarea(), a1 = f1.getarea();
	double l0 = sqrt(a0*a0), l1=sqrt(a1*a1);
	double area = (l0+l1)/2;
	double length = sqrt(e*e);
	a0=a0/l0;a1=a1/l1;
	double cosine = a0*a1, sine = cross(a0,a1)*(e/length);
	double theta = atan2(sine, cosine);
	return theta * length / area;
}

int parse(string x)
{
	int ans = 0;
	for (int i = 0; i < x.size(); ++i)
		if (x[i]<='9' && x[i]>='0')
			ans = ans * 10 + (x[i]-'0');
		else
			break;
	return ans-1;
}

void solve()
{
	ofstream fout;
	fout.open("log.txt",std::ofstream::out | std::ofstream::app);
	fout << ans << endl;
	fout.close();
}

void init()
{
	/*node.push_back(point(0,0,0));node.push_back(point(1,0,0));node.push_back(point(0,1,0));
	node.push_back(point(0.707,0,0.707));
	faces.push_back(face(0,1,2));faces.push_back(face(0,2,3));
	ans = calcurv(point(0,1,0),faces[0],faces[1]);
	return;*/
	ans = -1e10;
	for (int i = 0; i < 60; ++i)
	{
		node.clear();
		faces.clear();
		mesh.clear();
		string names = "";
		int x = i;
		while (x)
		{
			names = char(x%10+'0')+names;
			x /= 10;
		}
		while (names.size() < 4)
			names = "0" + names;
		ifstream fin(("./out/"+names+"_00.obj").c_str());
		string tmp;
		while (1)
		{
			getline(fin, tmp);
			if (tmp == "") break;
			if (tmp[0] == 'v' && tmp[1] == ' ')
			{
				stringstream sin(tmp.substr(2, tmp.size()-2));
				double x,y,z;
				sin >> x >> y >> z;
				node.push_back(point(x,y,z));
			}
			if (tmp[0]=='f' && tmp[1]==' ')
			{
				stringstream sin(tmp.substr(2, tmp.size()-2));
				string x,y,z;
				sin >> x >> y >> z;
				int p=parse(x),q=parse(y),r=parse(z);
				int num = faces.size();
				mesh[edge(p,q)].push_back(num);mesh[edge(q,r)].push_back(num);mesh[edge(p,r)].push_back(num);
				faces.push_back(face(p,q,r));
			}
		}
		fin.close();
		for (map<edge,vector<int> >::iterator it = mesh.begin(); it != mesh.end(); ++it)
		{
			if (it->second.size()<2) continue;
			double cur = calcurv(node[it->first.p]-node[it->first.q],faces[it->second[0]],faces[it->second[1]]);
			if (ans < cur)
			{
				ans = cur;
				cout << "\t" << ans << endl;
			}
		}
	}
}

int main()
{
	init();
	solve();
	return 0;
}