% clear
% % platemo('algorithm',@NSGAIII,'problem',@DTLZ1,'M',8,'D',12,'maxFE',78000,'N',156);
% for n_obj = [3,5,8,10,15]
%     k = 20;
%     n_var = n_obj + k - 1;
%     pro = DTLZ7('M',n_obj,'N',156,'D',n_var);
%     PF=pro.optimum;
%     dir="D:\PF_dtlz7_"+ n_obj+".txt"
%     writematrix(PF,dir)
% end
clear
dirPF="D:\D:\OneDrive\My Research\02_Finished\Many_Objectives\The Code\MaAVOA_Code\PF\PlatEmo";

listExp=GetSubDirsFirstLevelOnly("C:\Many_Objectives\EngProblem2");
PF_dir1="D:\OneDrive\My Research\02_Finished\Many_Objectives\The Code\MaAVOA_Code\PF\PF_EngProb1_4.txt";
optimum = table2array(readtable(PF_dir1));

for dir =listExp
    disp(dir);
    for i =1:400000
      F_dir="c:\Many_Objectives\EngProblem2"+"\"+dir+"\"+"run_"+i+"\F_new.csv";
      if isfile(F_dir)
          pat = digitsPattern;
          onlyNumbers = extract(dir, pat) ;
          metrics_dir="C:\Many_Objectives\EngProblem2"+"\"+dir+"\"+"run_"+i+"\metrics.csv";

          if isfile(metrics_dir) 
              disp(metrics_dir+" -- done")   ;         
          else
              disp(metrics_dir);
              p = table2array(readtable(F_dir));
              hv1=HV_new(p,optimum);
              igd1=IGD_new(p,optimum);
              gd1=GD_new(p,optimum);
              fid=fopen(metrics_dir,'w');
              fprintf(fid, "IGD ,GD ,igdplus ,HV\n");
              fprintf(fid, igd1+ " ,"+gd1+ " ,0 ,"+hv1+"\n");
          end
      end
    end
end