#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <cstdlib>
#include <float.h>
#include <random>
#include <stdint.h>
#include <climits>
#include <vector>
#include <queue>

#define _USE_MATH_DEFINES
using namespace std;

int NE = 300;
int NI = 100;
int Level = 100;//membrane potential
double PEE = 0.15;//probability of postsynaptic connections
double PIE = 0.5;
double PEI = 0.5;
double PII = 0.4;
double SEE = 4;//strength of postsynaptic connection
double SIE = 6.6/2.2;
double SEI = -2.2;
double SII = -2;
double kickE = 3000.0;//external drive rate
double kickI = 3000.0;
double Ref = 250.0;//time rate at state R
double HitE = 1000.0/2.0;//delay time rate
double HitI = 1000.0/3.0;//
int Reverse = -66;//reverse potential
int E_spike = 0;
int I_spike = 0;
double delay_start = 2.0/HitE;
double second_EE_start = 2.0/HitE;
double delay_end = 2.0/HitE;
double flag_off_time = 0.0000;
double flag_off_time_se = 0.000;
int start_threshold = 1;
int end_threshold = 2;
double MFE_interval = 0.0025;//interval between MFEs should be greater than this number
double MFE_min_length= 0.005;
double MFE_spike_count_threshold = 5;//number of spikes of a MFE should be greater than this number
double terminate_time = 7.0;

int real2int(const double x, mt19937& mt, uniform_real_distribution<double>& u)
//choose a random integer n if x is not an integer, such that the expectation of n is equal to x
{
    int xf = floor(x);
    double q = x - (double)xf;
    double y = 0;
    if( u(mt) < q)
        y = xf + 1;
    else
        y = xf;
    return y;
}

template <class T>
class Vector : public vector<T> {
private:
    T sum;
public:

    double get_sum()
    {
        return sum;
    }

    void maintain()
    {
        T tmp_sum = 0;
        for(auto && it : (*this))
            tmp_sum += it;
 //       cout<<tmp_sum<<endl;
        sum = tmp_sum;
    }

    void switch_element(const int index, T element)
    //switch [index] by element and maintain sum
    {
        sum = sum - (*this)[index] + element;
        (*this)[index] = element;
    }
    //The first three function is only for Clock[]

    bool remove_elt(unsigned index)
    //remove element [index]
    {
        if(this -> empty() || index < 0 || index >= this -> size()) {
            cout<<"Empty vector"<<endl;
            return false;
        }
        if(index == this->size() - 1) {
            this->pop_back();
        }
        else {
            T elt = (*this)[(*this).size() - 1];
            (*this)[index] = elt;
            (*this).pop_back();
        }
        return true;
    }

    int select(mt19937& mt, uniform_real_distribution<double>& u)
    {
        if( this->size() == 0)
        {
            cout<<"Size cannot be zero "<<endl;
            return -1;
        }
        else
        {
            int index = floor(u(mt)*this->size());
            T tmp =  (*this)[index];
            remove_elt(index);
            return tmp;
        }
    }

    int find_min() {
        if(!this -> empty()) {
            auto min = (*this)[0];
            auto index = 0;
            auto iter = 0;
            for(auto&& item : (*this)) {
                if(item < min) {
                    min = item;
                    index = iter;
                }
                iter++;
            }
            return index;
        }
        return -1;
    }

    void print_vector() {
        for(auto && it : (*this)) {
            cout<<it<<" ";
        }
        cout<<endl;
    }

};


int find_index(Vector<double>& array, mt19937& mt, uniform_real_distribution<double>& u)
//find a random element in a positive array. The probability of chosen is proportional to the value.
{
    double sum = array.get_sum();
    double tmp = u(mt);
    int index = 0;
    int size = array.size();
    double tmp_double = array[0]/sum;
//    cout<<tmp_double<<" "<<tmp<<endl;
    while(tmp_double < tmp && index < size - 1)
    {
        index++;
        tmp_double+=array[index]/sum;
    }
    return index;
}

void spikeE(const int whichHit, Vector<double>& Clock, vector<int> &VE, Vector<int> &HEE, Vector<int> &HIE, Vector<int> &Eref, vector<int> &awakeE, vector<int> &awakeI, mt19937& mt, uniform_real_distribution<double>& u)
{
    E_spike ++;
    VE[whichHit] = 0;
    awakeE[whichHit] = 0;
    Eref.push_back(whichHit);
    Clock.switch_element(6, Ref*Eref.size());
    for(int i = 0; i < NE; i++)
    {
        if(u(mt) < PEE && awakeE[i])
        {
            HEE.push_back(i);
        }
    }
    for(int i = 0; i < NI; i++)
    {
        if(u(mt) < PIE && awakeI[i])
        {
            HIE.push_back(i);
        }
    }
    Clock.switch_element(2, HitE*HEE.size());
    Clock.switch_element(3, HitE*HIE.size());

}
void spikeI(const int whichHit, Vector<double>& Clock, vector<int> &VI, Vector<int> &HEI, Vector<int> &HII, Vector<int> &Iref, vector<int> &awakeE, vector<int> &awakeI, mt19937& mt, uniform_real_distribution<double>& u)
{
    I_spike ++;
    VI[whichHit] = 0;
    awakeI[whichHit] = 0;
    Iref.push_back(whichHit);
    Clock.switch_element(7, Ref*Iref.size());
    for(int i = 0; i < NE; i++)
    {
        if(u(mt) < PEI && awakeE[i])
        {
            HEI.push_back(i);
        }
    }
    for(int i = 0; i < NI; i++)
    {
        if(u(mt) < PII && awakeI[i])
        {
            HII.push_back(i);
        }
    }
    Clock.switch_element(4, HitI*HEI.size());
    Clock.switch_element(5, HitI*HII.size());
}

void Histogram(vector<int>& VE, vector<int>& VI, vector<double>& Hist, const int start)
//Make a histogram at start*44-th entry of vector Hist
{
    int tmp = 0;
    for(auto i : VE)
    {
        tmp = (i+10)/5;
        if(tmp >= 0 && tmp < 22)
        Hist[start*44 + tmp]++;
    }
    for(auto i : VI)
    {
        tmp = (i+10)/5;
        if(tmp >= 0 && tmp < 22)
        Hist[start*44 + tmp + 22]++;
    }
}


void HistogramH(vector<int>& HEE, vector<int>& HIE, vector<int>& HEI, vector<int>& HII, vector<int>& HistH, const int start){
	for (int i = 0; i < (NE+NI)*2; i++){
		HistH[i] = 0;
	}
	for(auto i:HEE){
		HistH[start + (NE+NI)*0 + i]++;
	}
	for(auto i:HIE){
		HistH[start + (NE+NI)*0 + i + NE]++;
	}
	for(auto i:HEI){
		HistH[start + (NE+NI)*1 + i]++;
	}
	for(auto i:HII){
		HistH[start + (NE+NI)*1 + i + NE]++;
	}
}


void update(vector<double>& time_spike, vector<int>& num_spike, Vector<double>& Clock, vector<int> &VE, vector<int> &VI, Vector<int> &HEE, Vector<int> &HEI, Vector<int> &HIE, Vector<int> &HII, Vector<int> &Eref, Vector<int> &Iref, vector<int> &awakeE, vector<int> &awakeI, const double terminate_time, mt19937& mt, uniform_real_distribution<double>& u, vector<double>& X_train, vector<double>& Y_train)
{
    double current_time = 0.0;
    int count = 0;
    queue<double> wave_record;
	queue<double> wave_record_end;
    bool is_spike = false;
    double flag_time = 0.0;
    int flag = 0;//0 means no MFE, 1 means during the MFE
    int second_EE_flag=0; // 0 means no second EE 1 means yes
    vector<int> Wave_spike_count((int)(terminate_time*1000) );
    vector<double> MFE_time((int)(terminate_time*2000) );//2n entries: MFE start time; 2n+1 entries: MFE end time
    vector<double> second_EE_time ((int)(terminate_time*1000) );
    vector<double> HEE_stat((int)(terminate_time*3000));//3n entries: HEE at start time; 3n+1 entries: HEE at the peak; 3n+2 entries: HEE at end time
    vector<double> HIE_stat((int)(terminate_time*3000));
    vector<double> HEI_stat((int)(terminate_time*3000));
    vector<double> HII_stat((int)(terminate_time*3000));
    vector<int> E_ref((int)(terminate_time*2000));
    vector<int> I_ref((int)(terminate_time*2000));
    vector<double> X_train_local((int)(terminate_time*44000) );
    vector<double> Y_train_local((int)(terminate_time*44000) );
    int wave_count = 0;
    ofstream myfile, myfile1, myfile2, myfile3, myfile4, myfile5, myfile6, myfile7,myfile8;
    myfile.open("wave_info.txt");
    myfile1.open("pre_MFE_profile.txt");
    myfile2.open("post_MFE_profile.txt");
    myfile3.open("HEE_count.txt");
    myfile4.open("FullTrajectory15.txt");//pending spike count at the end of integer dt
    myfile5.open("H4_EEEIIEII_start.txt");
    myfile6.open("num_of_spike.txt");
    myfile7.open("ref_info.txt");// E start I start E end I end
    myfile8.open("H4_EEEIIEII_end.txt");
    double dt = 0.0005;
    double old_time = current_time;
    double HEE_max = 0.0;
    double HEI_max = 0.0;
    double HIE_max = 0.0;
    double HII_max = 0.0;

    vector<double> output_local(44);
    vector<int> HistH(2*(NE+NI));


    while(current_time < terminate_time)
    {
        old_time = current_time;
        current_time += -log(1 - u(mt))/Clock.get_sum();
        int step_start = (int)(old_time/dt);
        int step_end = (int)(current_time/dt);
        for(int j = step_start; j < step_end; j++)
        {
            //myfile4<<HEE.size()<<" "<<HEI.size()<<" "<<HIE.size()<<" "<<HII.size()<<endl;

            if(current_time > 5.0){

            for(int k = 0; k < NE ; k++){
            	myfile4<<VE[k]<<" ";
			}
			for(int k = 0; k < NI ; k++){
            	myfile4<<VI[k]<<" ";
			}
            HistogramH(HEE,HIE,HEI,HII, HistH, 0);
            for(int k=0;k<2 * (NE+NI); k++){
				myfile4<<HistH[k]<<" ";
			}
			for(int k = 0; k < NE ; k++){
            	myfile4<<awakeE[k]<<" ";
			}
			for(int k = 0; k < NI ; k++){
            	myfile4<<awakeI[k]<<" ";
			}
			myfile4<<"\n";
			}
        }
        is_spike = false;
        int index = find_index(Clock, mt, u);
        int whichHit;
        count ++;
        int local_index;
        switch (index)
        {
            case 0:
                whichHit = floor(u(mt)*NE);
                if(awakeE[whichHit])
                {
                    VE[whichHit]++;
                    if(VE[whichHit] >= Level)
                    {
                        spikeE(whichHit, Clock, VE, HEE, HIE, Eref, awakeE, awakeI, mt, u);
                        time_spike.push_back(current_time);
                        num_spike.push_back(whichHit);
                        if(flag == 1)
                        {
                            Wave_spike_count[wave_count]++;
                        }
                    }
                }
                break;
            case 1:
                whichHit = floor(u(mt)*NI);
                if(awakeI[whichHit])
                {
                    VI[whichHit]++;
                    if(VI[whichHit]>= Level)
                    {
                        spikeI(whichHit, Clock, VI, HEI, HII, Iref, awakeE, awakeI, mt, u);
                        time_spike.push_back(current_time);
                        num_spike.push_back(whichHit + NE);
                        if(flag == 1)
                        {
                            Wave_spike_count[wave_count]++;
                        }
                    }
                }
                break;
            case 2:
                whichHit = HEE.select(mt, u);
                if(awakeE[whichHit])
                {

                    VE[whichHit] += real2int(SEE, mt, u);
                    if( VE[whichHit] >= Level)
                    {
                        is_spike = true;
                        spikeE(whichHit, Clock, VE, HEE, HIE, Eref, awakeE, awakeI, mt, u);
                        time_spike.push_back(current_time);
                        num_spike.push_back(whichHit);
                        if(flag == 1)
                        {
                            Wave_spike_count[wave_count]++;
                        }
                    }
                }
                Clock.switch_element(2, HitE*HEE.size());
                break;
            case 3:
                whichHit = HIE.select(mt, u);
                if(awakeI[whichHit])
                {
                    VI[whichHit] += real2int(SIE, mt, u);
                    if( VI[whichHit] >= Level)
                    {
                        spikeI(whichHit, Clock, VI, HEI, HII, Iref, awakeE, awakeI, mt, u);
                        time_spike.push_back(current_time);
                        num_spike.push_back(whichHit + NE);
                        if(flag == 1)
                        {
                            Wave_spike_count[wave_count]++;
                        }
                    }
                }
                Clock.switch_element(3, HitE*HIE.size());
                break;
            case 4:
                whichHit = HEI.select(mt, u);
                if(awakeE[whichHit])
                {
                    VE[whichHit] += real2int(SEI, mt, u);
                    if(VE[whichHit] < Reverse)
                        VE[whichHit] = Reverse;
                }
                Clock.switch_element(4, HitI*HEI.size());
                break;
            case 5:
                whichHit = HII.select(mt, u);
                if(awakeI[whichHit])
                {
                    VI[whichHit] += real2int(SII, mt, u);
                    if(VI[whichHit] < Reverse)
                        VI[whichHit] = Reverse;
                }
                Clock.switch_element(5, HitI*HII.size());
                break;
            case 6:
                whichHit = Eref.select(mt, u);
                awakeE[whichHit] = 1;
                //printf("%d ",VE[whichHit]);
                Clock.switch_element(6, Ref*Eref.size());
                break;
            case 7:
                whichHit = Iref.select(mt, u);
                awakeI[whichHit] = 1;
                Clock.switch_element(7, Ref*Iref.size());
                break;
        }
        if(is_spike)
        {
            wave_record.push(current_time); wave_record_end.push(current_time);
        }
        while(!wave_record.empty() && current_time  - wave_record.front() > delay_start )
        {
            wave_record.pop();
        }
        while(!wave_record_end.empty() && current_time  - wave_record_end.front() > delay_end )
        {
            wave_record_end.pop();
        }

        if(flag==1 && is_spike && second_EE_flag==0){
			second_EE_time[wave_count]=current_time;
			//cout<<wave_count<<endl;
        	if(current_time - MFE_time[2*wave_count] > second_EE_start){

            	//flag=1;
            	//Start again at the same EE spike
            	flag_time = current_time;
            	for (int j = 0; j<44;j++) X_train_local[44*wave_count+j]=0;
            	Histogram(VE, VI, X_train_local, wave_count);
            	MFE_time[2*wave_count] = current_time;
            	HEE_stat[3*wave_count] = HEE.size();
            	HEE_max = HEE.size();
            	HEI_stat[3*wave_count] = HEI.size();
            	HEI_max = HEI.size();
            	HIE_stat[3*wave_count] = HIE.size();
            	HIE_max = HIE.size();
            	HII_stat[3*wave_count] = HII.size();
            	HII_max = HII.size();
            	E_ref[2*wave_count]=Eref.size();
                I_ref[2*wave_count]=Iref.size();
                Wave_spike_count[wave_count] = 0;

			}
			else{
				second_EE_flag=1;
			}
		}

		//if(flag == 0 && wave_record.size() >= start_threshold && current_time - flag_time >= flag_off_time)
        if(flag == 0 && is_spike && current_time - flag_time >= flag_off_time)
		{
            flag = 1;
            flag_time = current_time;
            Histogram(VE, VI, X_train_local, wave_count);
            MFE_time[2*wave_count] = current_time;
            HEE_stat[3*wave_count] = HEE.size();
            HEE_max = HEE.size();
            HEI_stat[3*wave_count] = HEI.size();
            HEI_max = HEI.size();
            HIE_stat[3*wave_count] = HIE.size();
            HIE_max = HIE.size();
            HII_stat[3*wave_count] = HII.size();
            HII_max = HII.size();

            E_ref[2*wave_count]=Eref.size();
            I_ref[2*wave_count]=Iref.size();
            Wave_spike_count[wave_count] = 0;

            second_EE_flag=0;
//            myfile<<current_time<<" "<<0<<endl;
			//cout<<current_time<<" "<<0<<endl;
        }

        if(flag == 1 )
        {
		    if(HEE_max < HEE.size())HEE_max = HEE.size();
		    if(HEI_max < HEI.size())HEI_max = HEI.size();
		    if(HIE_max < HIE.size())HIE_max = HIE.size();
		    if(HII_max < HII.size())HII_max = HII.size();
        }
        if(flag == 1 && second_EE_flag==1 && current_time - flag_time >= flag_off_time_se &&( (wave_record_end.size() <= end_threshold ) || ( current_time > flag_time + 0.01  && HEE.size() < 50 ) ) )
        {
            flag = 0;
            second_EE_flag=0;
            flag_time = current_time;
            Histogram(VE, VI, Y_train_local, wave_count);
            MFE_time[2*wave_count + 1] = current_time;
            HEE_stat[3*wave_count + 1] = HEE_max;
            HEE_stat[3*wave_count + 2] = HEE.size();

            HEI_stat[3*wave_count + 1] = HEI_max;
            HEI_stat[3*wave_count + 2] = HEI.size();

            HIE_stat[3*wave_count + 1] = HIE_max;
            HIE_stat[3*wave_count + 2] = HIE.size();

            HII_stat[3*wave_count + 1] = HII_max;
           	HII_stat[3*wave_count + 2] = HII.size();

           	E_ref[2*wave_count+1]=Eref.size();
            I_ref[2*wave_count+1]=Iref.size();
           	/*
           	printf("%d -- ",Eref.size());
			for(auto i : Eref)printf("%d ",i);
           	printf("\n V: ");
			for(auto i : Eref)printf("%d ",VE[i]);
           	printf("\n");
           	*/
            HEE_max = 0.0;
            HEI_max = 0.0;
            HIE_max = 0.0;
            HII_max = 0.0;
            wave_count++;
            //myfile<<current_time<<" "<<1<<endl;
			//cout<<current_time<<" "<<1<<endl;
        }
		//cout<<"ok "<<current_time<<endl;
    }

    //merge MFEs

    int index2 = 0;
    int index = 0;
    vector<int> Wave_spike_count2(wave_count);
    vector<double> MFE_time2(2*wave_count );//2n entries: MFE start time; 2n+1 entries: MFE end time
    vector<double> HEE_stat2(3*wave_count );//3n entries: HEE at start time; 3n+1 entries: HEE at the peak; 3n+2 entries: HEE at end time
    vector<double> HEI_stat2(3*wave_count );
    vector<double> HIE_stat2(3*wave_count );
    vector<double> HII_stat2(3*wave_count );
    vector<int> E_ref2(2*wave_count);
	vector<int> I_ref2(2*wave_count);
	vector<int> X2(44*wave_count);
	vector<int> Y2(44*wave_count);
    vector<double> second_EE_time2(wave_count);
	/*
	for(auto a : HEE_stat)
        cout<<a<<" ";
    cout<<endl;
    */
    while(index < wave_count - 1)
    {
        MFE_time2[2*index2] = MFE_time[2*index];
        second_EE_time2[index2]=second_EE_time[index];
        HEE_stat2[3*index2] = HEE_stat[3*index];
        HEI_stat2[3*index2] = HEI_stat[3*index];
        HIE_stat2[3*index2] = HIE_stat[3*index];
        HII_stat2[3*index2] = HII_stat[3*index];
        E_ref2[2*index2] = E_ref[2*index];
        I_ref2[2*index2] = I_ref[2*index];
        for(int j=0;j<44;j++){
        	X2[44*index2+j]=X_train_local[44*index+j];
		}

        int local_sp_count = 0;
        int local_HEE_max = HEE_stat[3*index + 1];
        int local_HEI_max = HEI_stat[3*index + 1];
        int local_HIE_max = HIE_stat[3*index + 1];
        int local_HII_max = HII_stat[3*index + 1];
        do
        {
            local_sp_count += Wave_spike_count[index];
            if(local_HEE_max < HEE_stat[3*index + 1])
            {
                local_HEE_max = HEE_stat[3*index + 1];
            }
            if(local_HEI_max < HEI_stat[3*index + 1])
            {
                local_HEI_max = HEI_stat[3*index + 1];
            }
            if(local_HIE_max < HIE_stat[3*index + 1])
            {
                local_HIE_max = HIE_stat[3*index + 1];
            }
            if(local_HII_max < HII_stat[3*index + 1])
            {
                local_HII_max = HII_stat[3*index + 1];
            }
            HEE_stat2[3*index2 + 2] = HEE_stat[3*index + 2];
            HEI_stat2[3*index2 + 2] = HEI_stat[3*index + 2];
            HIE_stat2[3*index2 + 2] = HIE_stat[3*index + 2];
            HII_stat2[3*index2 + 2] = HII_stat[3*index + 2];
			MFE_time2[2*index2 + 1] = MFE_time[2*index + 1];
			E_ref2[2*index2+1] = E_ref[2*index+1];
        	I_ref2[2*index2+1] = I_ref[2*index+1];
        	for(int j=0;j<44;j++){
        		Y2[44*index2+j]=Y_train_local[44*index+j];
			}
            index++;
        }
        while(MFE_time[2*index] - MFE_time[2*index - 1] < MFE_interval);
        Wave_spike_count2[index2] = local_sp_count;
        HEE_stat2[3*index2 + 1] = local_HEE_max;
        HEI_stat2[3*index2 + 1] = local_HEI_max;
        HIE_stat2[3*index2 + 1] = local_HIE_max;
        HII_stat2[3*index2 + 1] = local_HII_max;
        index2 ++;
    }
    wave_count = index2;
    for(int i = 0; i < wave_count; i++)
    {
        MFE_time[2*i] = MFE_time2[2*i];
        MFE_time[2*i+1] = MFE_time2[2*i+1];
        HEE_stat[3*i] = HEE_stat2[3*i];
        HEE_stat[3*i + 1] = HEE_stat2[3*i + 1];
        HEE_stat[3*i + 2] = HEE_stat2[3*i + 2];

        HEI_stat[3*i] = HEI_stat2[3*i];
        HEI_stat[3*i + 1] = HEI_stat2[3*i + 1];
        HEI_stat[3*i + 2] = HEI_stat2[3*i + 2];

        HIE_stat[3*i] = HIE_stat2[3*i];
        HIE_stat[3*i + 1] = HIE_stat2[3*i + 1];
        HIE_stat[3*i + 2] = HIE_stat2[3*i + 2];

        HII_stat[3*i] = HII_stat2[3*i];
        HII_stat[3*i + 1] = HII_stat2[3*i + 1];
        HII_stat[3*i + 2] = HII_stat2[3*i + 2];

        E_ref[2*i] = E_ref2[2*i];
        E_ref[2*i+1] = E_ref2[2*i+1];
        I_ref[2*i] = I_ref2[2*i];
        I_ref[2*i+1] = I_ref2[2*i+1];

        for(int j=0;j<44;j++){

        	X_train_local[44*i+j]=X2[44*i+j];
        	Y_train_local[44*i+j]=Y2[44*i+j];

    	}

        Wave_spike_count[i] = Wave_spike_count2[i];
        second_EE_time[i]=second_EE_time2[i];
    }
    second_EE_time.resize(wave_count);
    MFE_time.resize(2*wave_count);
    HEE_stat.resize(3*wave_count);
    HEI_stat.resize(3*wave_count);
    HIE_stat.resize(3*wave_count);
    HII_stat.resize(3*wave_count);
    E_ref.resize(2*wave_count);
    I_ref.resize(2*wave_count);
    Wave_spike_count.resize(wave_count);

    //finish merging MFEs


    X_train_local.resize(44*(wave_count));
    Y_train_local.resize(44*(wave_count));
    int eff_MFE_counter = 0;
    for(int i = 0; i < wave_count; i++)
    {
        cout<<endl;
        cout<<" Wave Id = "<<i<<" ";
        cout<<"MFE starts at "<<MFE_time[2*i]<<" ends at "<<MFE_time[2*i+1]<<endl;
        cout<<"number of spikes = "<<Wave_spike_count[i]<<endl;
        cout<<"HEE start = "<<HEE_stat[3*i]<<" HEE peak = "<<HEE_stat[3*i + 1]<<" HEE end = "<<HEE_stat[3*i + 2]<<endl;
        cout<<"pre MFE profile = ";
        if(i > 5 && second_EE_time[i]-MFE_time[2*i]<=second_EE_start && MFE_time[2*i+1]>MFE_time[2*i]+MFE_min_length && Wave_spike_count[i] > MFE_spike_count_threshold && HEE_stat[3*i+1] > 100 + HEE_stat[3*i])
        {
        	//cout<<i<<endl;
            for(int j = 0; j < 44; j++)
            {
                if(j == 0)
                    cout<<" E: ";
                if(j == 22)
                    cout<<" I: ";
                cout<<X_train_local[i*44 + j]<<" ";
                X_train[eff_MFE_counter*44+j] = X_train_local[i*44+j];
                myfile1<<X_train[eff_MFE_counter*44+j]<<" ";
            }
            myfile1<<endl;
            cout<<endl;
            myfile<<MFE_time[2*i]<<" "<<0<<endl;
            myfile<<MFE_time[2*i+1]<<" "<<1<<endl;
            cout<<"post MFE profile = ";
            for(int j = 0; j < 44; j++)
            {
                if(j == 0)
                    cout<<" E: ";
                if(j == 22)
                    cout<<" I: ";
                cout<<Y_train_local[i*44 + j]<<" ";
                Y_train[eff_MFE_counter*44+j] = Y_train_local[i*44+j];
                myfile2<<Y_train[eff_MFE_counter*44+j]<<" ";
            }
            cout<<endl;
            myfile2<<endl;
            eff_MFE_counter++;
            myfile3<<HEE_stat[3*i]<<" "<<HEE_stat[3*i+1]<<" "<<HEE_stat[3*i+2]<<endl;
            myfile5<<HEE_stat[3*i]<<" "<<HEI_stat[3*i]<<" "<<HIE_stat[3*i]<<" "<<HII_stat[3*i]<<endl;
            myfile8<<HEE_stat[3*i+2]<<" "<<HEI_stat[3*i+2]<<" "<<HIE_stat[3*i+2]<<" "<<HII_stat[3*i+2]<<endl;
            myfile6<<Wave_spike_count[i]<<endl;
            myfile7<<E_ref[2*i]<<" "<<I_ref[2*i]<<" "<<E_ref[2*i+1]<<" "<<I_ref[2*i+1]<<endl;
        }
        else
        {
            cout<<"MFE is not counted"<<endl;
        }
    }
    //printf("ok");
    X_train.resize(44*(eff_MFE_counter - 1));
    Y_train.resize(44*(eff_MFE_counter - 1));
    myfile.close();
    myfile1.close();
    myfile2.close();
    myfile3.close();
    myfile4.close();
    myfile5.close();
    myfile6.close();
    myfile7.close();
    myfile8.close();
}

int main()
{
    struct timeval t1, t2;
    gettimeofday(&t1,NULL);
    ofstream myfile;
    random_device rd;
    mt19937 mt(11);
    uniform_real_distribution<double> u(0, 1);
    myfile.open("spike_info.txt");
    vector<int> VE(NE);//membrane potential
    vector<int> VI(NI);
    for(auto i : VE)
        i = 0;
    for(auto i : VI)
        i = 0;
    Vector<int> HEE, HEI, HII, HIE, Eref, Iref;
    //HEE, HEI ... pending postsynaptic kick
    //Eref, Iref: indices of neurons at state R
    HEE.reserve(100000);
    HEI.reserve(100000);
    HII.reserve(100000);
    HIE.reserve(100000);
    Eref.reserve(NE);
    Iref.reserve(NI);
    vector<int> awakeE(NE);
    for(auto & i : awakeE)
        i = 1;
    vector<int> awakeI(NI);
    for(auto & i : awakeI)
        i = 1;
    Vector<double> Clock;
    Clock.reserve(8);
    //0 Edrive, 1 Idrive, 2 HEE, 3, HEI, 4, HIE, 5, HII, 6, Eref, 7, Iref
    Clock.push_back(NE*kickE);
    Clock.push_back(NI*kickI);
    for(int i = 2; i < 8; i++)
        Clock.push_back(0);
    Clock.maintain();
    //double terminate_time = 2.0;
    vector<double> time_spike;
    time_spike.reserve(100000);
    vector<int> num_spike;
    num_spike.reserve(100000);
    vector<double> X_train(int(44*1000*terminate_time));
    vector<double> Y_train(int(44*1000*terminate_time));
    update(time_spike, num_spike, Clock, VE, VI, HEE, HEI, HIE, HII, Eref, Iref, awakeE, awakeI, terminate_time, mt, u, X_train, Y_train);

    cout<<"E spike rate= "<<(double)E_spike/(terminate_time*NE)<<endl;
    cout<<"I spike rate = "<<(double)I_spike/(terminate_time*NI)<<endl;
    int spike_count = time_spike.size();
    /*for(int i = 0; i < spike_count; i++)
    {
        myfile<<time_spike[i]<<"  "<<num_spike[i]<<endl;
    }*/
    myfile.close();


    gettimeofday(&t2, NULL);
    double delta = ((t2.tv_sec  - t1.tv_sec) * 1000000u +
                    t2.tv_usec - t1.tv_usec) / 1.e6;

    cout << "total CPU time = " << delta <<endl;

}







