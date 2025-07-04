%%%% Thesis style for Khalifa University, developed by:
%           Dr. Maha Kadadha
%           Eman Ouda
%           Dr. Andrei Sleptchenko

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{KUThesis}[2022/09/22 PhD/Master Thesis class for Khalifa University]
\RequirePackage{ifthen}
\RequirePackage{xifthen}


\newboolean{phdthesis}
\newboolean{masterthesis}

\newboolean{singlespacingbl}
\newboolean{onehalfspacingbl}
\newboolean{doublespacingbl}

% default single spacing
\setboolean{singlespacingbl}{true}
\setboolean{onehalfspacingbl}{false} 
\setboolean{doublespacingbl}{false}

\newboolean{proposalbl}
\setboolean{proposalbl}{false}


% \newcommand{\headlinecolor}{\normalcolor}
% \RequirePackage{xcolor}
% \definecolor{slcolor}{HTML}{882B21}

\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{msc}{ \setboolean{phdthesis}{false} \setboolean{masterthesis}{true}}
\DeclareOption{phd}{ \setboolean{phdthesis}{true} \setboolean{masterthesis}{false}}
\DeclareOption{proposal}{ \setboolean{proposalbl}{true}}
% \DeclareOption{green}{\renewcommand{\headlinecolor}{\color{green}}}
% \DeclareOption{red}{\renewcommand{\headlinecolor}{\color{slcolor}}}
\DeclareOption{singlespace}{  \setboolean{singlespacingbl}{true}  \setboolean{onehalfspacingbl}{false} \setboolean{doublespacingbl}{false} }
\DeclareOption{onehalfspace}{ \setboolean{singlespacingbl}{false} \setboolean{onehalfspacingbl}{true}  \setboolean{doublespacingbl}{false} }
\DeclareOption{doublespace}{  \setboolean{singlespacingbl}{false} \setboolean{onehalfspacingbl}{false} \setboolean{doublespacingbl}{true} }
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[a4paper,12pt]{report}





%{{ Don't change! times font for proper scaling
\RequirePackage{mathptmx}
\RequirePackage[scaled=.90]{helvet}
\RequirePackage{courier}
%}} times font for proper scaling
\RequirePackage{caption}
% \RequirePackage{subcaption}
\RequirePackage[T1]{fontenc}
\RequirePackage[numbers,square]{natbib} %see http://merkel.zoneo.net/Latex/natbib.php for usage!
\RequirePackage[pdftex]{graphicx}
\RequirePackage{tikz}
\RequirePackage{xtab, tabularx} %for multipage tables
\RequirePackage[Sonny]{fncychap} %for the chapter headings
\RequirePackage{makeidx}
\RequirePackage{float}
\RequirePackage{amsmath}
\usepackage{amsfonts}
%{{ set double-space
\RequirePackage{setspace}
%\doublespacing
%}} set double-space
\RequirePackage[pdftex]{hyperref}
\RequirePackage{etoolbox}
\RequirePackage{makecell}

\hypersetup{%
  pdfcreator = {\LaTeX\ with package \flqq hyperref\frqq},
  pdfproducer = {pdfeTeX-0.\the\pdftexversion\pdftexrevision},
  pdfstartview=FitH, % Fit the page horizontally
%  bookmarks=true, % Open Bookmarks in AR
  colorlinks=true,
  linkcolor=blue,
  anchorcolor=blue,
  citecolor=blue,
  menucolor=blue,
  urlcolor=blue,
  %pagecolor=blue,
  filecolor=blue,
 % backref=false,
  %letterpaper=true,
  breaklinks=true,
}


%Toc
\RequirePackage{titletoc}
\titlecontents{chapter}
  [0pt]                                  % Left margin
  {\addvspace{1em}}                      % Space before chapter title
  {\bfseries\color{blue}{\chaptername}\ \thecontentslabel.\quad} % Chapter label
  {\bfseries}                            % Chapter title
  {\hfill\bfseries\contentspage\vspace{0.5em}}


\AfterPreamble{
    % generates list of keywords for pdf 
    \makeatletter
    \ifnum\theidxtrmcnt>0
        \toks@={\getidxtrm{1}}
        \@tempcnta=\@ne
        \loop\ifnum\@tempcnta<\theidxtrmcnt\relax
            \advance\@tempcnta\@ne
            \toks@=\expandafter{\the\expandafter\toks@\expandafter,\expandafter~\expandafter\getidxtrm\expandafter{\number\expandafter\@tempcnta\expandafter}}
        \repeat
        \edef\keywords{{\the\toks@}}%
    \else
        \edef\keywords{}%
    \fi
    \makeatother
}        
\AtBeginDocument{
    \hypersetup{
        pdfauthor={\@authorshort},
        pdftitle={\@titleshort},
        pdfsubject={Graduation thesis/dissertation for \@studyprogram},        
        pdfkeywords={Khalifa Univesity, thesis/dissertation, \keywords}
    }
}


%set margins
\RequirePackage[top=1in, bottom=1in, left=1.25in, right=1.25in]{geometry}

\RequirePackage{datetime}

\newdateformat{monthyeardate}{%
  \monthname[\THEMONTH], \THEYEAR}
%{{ Fill in proper metadata!!! and reference highlighting for the generated PDF file


\renewcommand*{\author}[2][]{%
  \gdef\@authorlong{#2}
  \ifstrequal{#1}{}
    {\gdef\@authorshort{#2}}
    {\gdef\@authorshort{#1}}
  % \gdef\@authorshort{\ifstrequal{#1}{}{#2}{#1}
}
\def\@authorlong{%
  \ClassError{KUThesis}{No author name given.}%
}

\renewcommand*{\title}[2][]{%
  \gdef\@titlelong{#2}
  \ifstrequal{#1}{}
    {\gdef\@titleshort{#2}}
    {\gdef\@titleshort{#1}}
  % \gdef\@titleshort{\ifempty{#1} #2 \else #1 \fi}
}
\def\@titlelong{%
  \ClassError{KUThesis}{No author name given.}%
}

\newcommand*{\department}[1]{%
  \gdef\@department{#1}
}
\def\@department{%
  \ClassError{KUThesis}{No department given.}%
}

\newcommand*{\studyprogram}[1]{%
  \gdef\@studyprogram{#1}
}
\def\@studyprogram{%
  \ClassError{KUThesis}{No program name given.}%
}

\renewcommand*{\abstract}[1]{%
  \gdef\@abstracttext{#1}
}
\def\@abstracttext{%
  \ClassError{KUThesis}{No abstract provided.}%
}

\newcommand*{\acknowledgment}[1]{%
  \gdef\@acknowledgmenttext{#1}
}
\def\@acknowledgmenttext{%
  \ClassError{KUThesis}{No acknowledgement provided.}%
}


\newcommand{\currentdate}{\ifcase \month \or January\or February\or March\or %
April\or May \or June\or July\or August\or September\or October\or November\or %
December\fi, \number \year} 

\newcounter{rsccnt}
\newcommand\rsclist{}
\newcommand\setrscmember[2]{%
  \csdef{rsc#1}{#2}}
\newcommand\addrscmember[3]{%
	\stepcounter{rsccnt}%
	\csdef{rsc\thersccnt}{\ifstrequal{#1}{}%
	    {}% #1 = #2 -> [#1]
	    {\makecell[l]{#1 (#2), \\ \textit{\small#3}}}% [#1,#2]
	   } }
\newcommand\getrscmember[1]{%
  \csuse{rsc#1}}
  
  

\newcounter{idxtrmcnt}
\newcommand\idxtrmlist{}
\newcommand\setidxtrm[2]{%
  \csdef{term#1}{#2}}
\newcommand\addidxtrm[1]{%
	\stepcounter{idxtrmcnt}%
	\csdef{term\theidxtrmcnt}{#1}}
\newcommand\getidxtrm[1]{%
  \csuse{term#1}}

\AfterPreamble{
    \def\keywords{
        \newcounter{i}
        \loop
        \stepcounter{i}
        \getidxtrm{\thei},\
        \ifnum \value{i}<\value{idxtrmcnt}
        \repeat
    }
}

\RequirePackage{nomencl}
\renewcommand{\nomname}{List of Abbreviations and Notations}
% \renewcommand{\nompreamble}{The next list describes  abbreviations and notations that will be later used within the body of the document}

%% This code creates the groups
% -----------------------------------------
\renewcommand\nomgroup[1]{%
  \item[\bfseries
  \ifstrequal{#1}{A}{Abbreviations}{%
  \ifstrequal{#1}{N}{Notations}}%
]}
% -----------------------------------------
\makenomenclature



%----------------------------------------------------------------------
\makeindex % activate index-making


%----------------------------------------------------------------------
% Define your own commands and environment definitions, for example:
\newcommand{\program}[1]{\textbf{#1}} % program names in bold text
\newcommand{\exten}[1]{\texttt{#1}} % file extensions in bold text (use caps)
\newcommand{\cmmd}[1]{\textbackslash\texttt{#1}} % command name in tt font
\newcommand{\enviro}[1]{\texttt{#1}} % environment name in tt font


\newenvironment{definition}[1]{\begin{quote}\emph{#1}:}{\end{quote}}
  % Provides indented formal definition and emphasizes the word.
  % e.g. \begin{definition}{Reliability} ... \end{definition}


\newcommand{\makecoverpages}{
  \thispagestyle{empty}
  \begin{titlepage}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \setcounter{footnote}{0}

    \begin{center}
      	\includegraphics[width=0.6\textwidth]{logos/KUlogoBlue.png}
		\vspace{\stretch{0.8}}

		\begin{spacing}{2.5}
			\fontsize{28}{30}{\selectfont\textbf{ \@titlelong}}
		\end{spacing}
		\vspace{\stretch{1}}

       \fontsize{22}{24}{\selectfont \@authorlong }
		\vspace{\stretch{1}}


      \fontsize{20}{24}{\selectfont   
	      \ifphdthesis
	      		Ph.D. Dissertation
	      \else 
		      \ifmasterthesis
	    	  		MSc. Thesis
	    	  \else
	    	  		Thesis
	    	  \fi
	     \fi
          \ifproposalbl Proposal  \fi
       }
		\vspace{\stretch{1}}

	\fontsize{18}{24}{\selectfont \@date}
		\vspace{\stretch{0.6}}
    \end{center}
    
    \begin{spacing}{1.4}
        \fontsize{11}{12}{\selectfont{ 
            \noindent
                A  \ifphdthesis  dissertation \else  thesis  \fi \ifproposalbl proposal  \fi
                submitted to Khalifa University of Science and Technology in accordance with
                the requirements of the degree of \@studyprogram, in the Department of \@department }}
    \end{spacing}

    \vspace{\stretch{0.5}}
    

 
  \end{titlepage}
  

%--------------------------------------------------------------------------
\newpage
\thispagestyle{empty}
\ifproposalbl
        \fontsize{14}{24}{\selectfont \bf Proposal Examination Committee}
		\bigskip\bigskip\bigskip
		
		\begin{tabular}{p{6cm}p{6cm}}
			\getrscmember{1} & 			\getrscmember{5} \\[2pt]
			\getrscmember{2} & 			\getrscmember{6} \\[2pt]
			\getrscmember{3} & 			\getrscmember{7} \\[2pt]
			\getrscmember{4} & 			\getrscmember{8} 
		\end{tabular}		

\else
	\begin{center}
	
		\vspace*{1.5ex}
		\includegraphics[width=.6\textwidth]{logos/KUlogoBlue.png}
		\vspace{\stretch{1}}
		%\bigskip\bigskip\bigskip
		
		\begin{spacing}{2}
			\fontsize{22}{24}{\selectfont\textbf{ \@titlelong}}
%   			\{\fontsize{22}{24}\selectfont	\textbf{Full Dissertation Title in 22 PT, Bold, Times New Roman}}
		\end{spacing}

		\vspace{\stretch{0.5}}
		\fontsize{14}{14}{\selectfont by}
		\bigskip
		
        \fontsize{18}{24}{\selectfont \@authorlong }
%		\fontsize{18}{24}{\selectfont  Student Name, 18 PT, Times New Roman \par}
		\vspace{\stretch{1}}
		
		\fontsize{14}{24}{\selectfont A  \ifphdthesis  Dissertation \else  Thesis  \fi 
							submitted in partial fulfillment of the \\requirements for the degree of} % (14pt)
		\vspace{\stretch{1}}
		
		\fontsize{16}{24}{\selectfont \bf \@studyprogram}  % Name - Concentration, 16 PT, Times New Roman \par}
		\bigskip\bigskip
		
		\fontsize{14}{24}{\selectfont at}
		\bigskip\smallskip
		
		\fontsize{20}{24}{\selectfont  Khalifa University}
		\vspace{\stretch{1}}
		

		\fontsize{14}{24}{\selectfont \bf \ifphdthesis  Dissertation Defense \else  Thesis  \fi Committee}
		\bigskip\bigskip\bigskip
		
		\begin{tabular}{p{6cm}p{6cm}}
			\getrscmember{1} & 			\getrscmember{5} \\[2pt]
			\getrscmember{2} & 			\getrscmember{6} \\[2pt]
			\getrscmember{3} & 			\getrscmember{7} \\[2pt]
			\getrscmember{4} & 			\getrscmember{8} 
		\end{tabular}		
		
		\bigskip\bigskip\bigskip\bigskip
	\fontsize{14}{24}{\selectfont \@date} %18 PT, Times New Roman  \par}
		
	
	
	\end{center}
 \fi
%	\clearpage

}


\newcommand{\makefrontpages}{
	\newpage
	% PRELIMINARY PAGES
	\pagestyle{plain} % No headers, just page numbers
	\pagenumbering{roman} % Roman numerals
	\setcounter{page}{3}
	
%    \renewcommand{\chaptermark}[1]{ \markboth{#1}{#1}}
	
	%%%%%%%%%%%%%%%%  Abstract   %%%%%%%%%%%%%%
	  %\textbf{\Huge  Abstract} 		
	\paragraph{}\addcontentsline{toc}{section}{Abstract}
    \noindent\hskip-1em
    \fontsize{23}{24}{\selectfont\bf Abstract}
	\bigskip\bigskip
	
	\begin{singlespace}
	 \noindent \@authorlong, \textbf{``\@titlelong''}, 
	 \ifphdthesis  PhD Dissertation\else  M.Sc. Thesis\fi\ifproposalbl\ Proposal\fi, 
	 \@studyprogram, 
	 Department of \@department, 
	 Khalifa University of Science and Technology, United Arab Emirates, 
	 \@date.
	 \bigskip
	
	\noindent\@abstracttext
	
	
	 \bigskip
	 \noindent\textbf{Indexing Terms: } \foreach \i in {1,...,\theidxtrmcnt}{\getidxtrm{\i};\ }
     

	\end{singlespace}

	\ifproposalbl
    \else
    	%%%%%%%%%%%%%%%%  Acknowledgments   %%%%%%%%%%%%%%
    	\newpage
    	\paragraph{}\addcontentsline{toc}{section}{Acknowledgments}
        \noindent\hskip-1em
    	\fontsize{23}{24}{\selectfont \bf Acknowledgments}
    	\bigskip\bigskip
    	
    	\begin{singlespace}
    	\normalsize
    	\noindent\@acknowledgmenttext
    	\end{singlespace}
    	
    	
    	%%%%%%%%%%%%%%%%     Declaration and Copyright    %%%%
    	\newpage
    	\paragraph{}\addcontentsline{toc}{section}{Declaration and Copyright}
        \noindent\hskip-1em
    	\fontsize{23}{24}{\selectfont \bf Declaration and Copyright}
     	\bigskip\bigskip\bigskip
     	
    	
    	\begin{singlespace}
    		\addtocontents{toc}{\vspace{1em}}  % Add a gap in the Contents, for aesthetics
    		\bigskip	\bigskip
    		\noindent %\textbf{\LARGE Declaration}
    		\fontsize{19}{24}{\selectfont \bf Declaration}
    		\bigskip	\bigskip \bigskip
    		
    		\noindent I declare that the work in this \ifphdthesis  dissertation \else  thesis  \fi  was carried out in accordance with the regulations of Khalifa University of Science and Technology. 
    		The work is entirely my own except where indicated by special reference in the text. Any views expressed in this \ifphdthesis  dissertation \else  thesis  \fi  are those of the author and in no way represent those of Khalifa University of Science and Technology. 
    		No part of this \ifphdthesis  dissertation \else  thesis  \fi  has been presented to any other university for any degree.
    	\end{singlespace}
    	\bigskip
    	\bigskip	
    	\noindent Author Name:\\
    	\phantom{Author Name:\ }\rule[1em]{25em}{0.7pt}  % This prints a line for the signature
    	\bigskip	
    	
    	\noindent Author Signature:\\
    	\phantom{Author Signature:\ }\rule[1em]{23.55em}{0.7pt}  % This prints a line for the signature
    	\bigskip	
    	
    	\noindent Date:\\
    	\phantom{Date:\ }\rule[1em]{28.56em}{0.7pt}  % This prints a line to write the date
    	
    
    	\bigskip\bigskip\bigskip\bigskip
    	\bigskip\bigskip\bigskip\bigskip
    	\noindent\fontsize{19}{24}{\selectfont \bf Copyright \textcopyright}
    	\bigskip	\bigskip	\bigskip \bigskip
    	\begin{singlespace}
    		\noindent No part of this \ifphdthesis  dissertation \else  thesis  \fi may be reproduced, stored in a retrieval system, or transmitted, in any form or by any means, electronic, mechanical, photocopying, recording, scanning or otherwise, without prior written permission of the author. The \ifphdthesis  dissertation \else  thesis  \fi may be made available for consultation in Khalifa University of Science and Technology Library and for inter-library lending for use in another library and may be copied in full or in part for any bona fide library or research worker, on the understanding that users are made aware of their obligations under copyright, i.e. that no quotation and no information derived from it may be published without the author's prior consent.
    	\end{singlespace}
    \fi		

	%% ----------------------------------------------------------------
	
	
	
	%%%%%%%%%%%%%%%%%%%%%
	% Pages which are generated automatically
	%\setcounter{page}{11} % Set this counter to get correct page numbers
	\normalsize
	\tableofcontents\addcontentsline{toc}{section}{Contents}
	
	
	\listoffigures\addcontentsline{toc}{section}{List of Figures}
	
	
	\listoftables\addcontentsline{toc}{section}{List of Tables}


	\printnomenclature
	\addcontentsline{toc}{section}{\nomname}

%                \renewcommand{\chaptermark}[1]{%
%                                \markboth{\chaptername\ \thechapter.\ #1}{}}



% \ifpdf
%     \pdfinfo{
%        /Author (\\@authorlong)
%        /Title  (\\@titlelong)
%        /CreationDate ()
%        /ModDate (D:\pdfdate)
%        /Subject (Graduation thesis/dissertation for \\@studyprogram)
%        /Keywords (\\foreach \\i in {1,...,\\theidxtrmcnt}{\\getidxtrm{\i}})
%     }
% \fi

    \clearpage
	\pagestyle{headings}
	% Go to normal sized type
	\normalsize
	\setcounter{page}{1} %ChangePageNumber
	\pagenumbering{arabic} % back to arabic number

    
    \ifsinglespacingbl \singlespacing \else
        \ifonehalfspacingbl \onehalfspacing \else
            \ifdoublespacingbl \doublespacing \fi \fi \fi




}