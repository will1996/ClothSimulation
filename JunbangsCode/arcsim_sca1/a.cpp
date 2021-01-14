#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>
using namespace std;

void doit(string a, string b)
{
	ifstream fin(a.c_str());
	ofstream fout(b.c_str());
	int cnt = 0;
	string s;
	while (getline(fin, s))
	{
		++cnt;
		if (cnt > 3)
			fout << s << endl;
	}
	fin.close();
	fout.close();
}

string stringf(string a, int x)
{
	a = a + char('0'+x/10);
	a = a + char('0'+x%10);
	return a;
}

int main()
{
	for (int i = 0; i <= 70; ++i)
		doit(stringf("./obs/1_0000", i)+".obj",stringf("./obs/2_0000", i)+".obj");
	return 0;
}