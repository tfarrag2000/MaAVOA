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
problist=["dtlz1","dtlz2","dtlz3","dtlz4","dtlz5","dtlz6","dtlz7"];
dirPF="..\MaAVOA_Code\PF\PlatEmo";

listExp=GetSubDirsFirstLevelOnly("C:\Many_Objectives\DTLZ_Problems");

for dir =listExp
    disp(dir);
    for i =1:40
      F_dir="C:\Many_Objectives\DTLZ_Problems"+"\"+dir+"\"+"run_"+i+"\F_new.csv";
      if isfile(F_dir)
          pat = digitsPattern;
          onlyNumbers = extract(dir, pat) ;
          PF_dir1=dirPF+"\PF_dtlz"+onlyNumbers(1) +"_"+onlyNumbers(2)+".txt";
          metrics_dir="C:\Many_Objectives\DTLZ_Problems"+"\"+dir+"\"+"run_"+i+"\metrics.csv";
          disp(metrics_dir);

          if isfile(metrics_dir) 
              disp("done")   ;         
          else
              disp(metrics_dir);
              p = table2array(readtable(F_dir));
              optimum = table2array(readtable(PF_dir1));
              hv1=HV_new(p,optimum);
              igd1=IGD_new(p,optimum);
              gd1=GD_new(p,optimum);
              fid=fopen(metrics_dir,'w');
              fprintf(fid, "IGD ,GD ,igdplus ,HV\n");
              fprintf(fid, igd1+ " ,"+gd1+ " ,0 ,"+hv1+"\n");
              fclose(fid);

          end
      end
    end
end