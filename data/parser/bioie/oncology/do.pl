#!/usr/bin/perl -w

open(F1,"ls source*|");
foreach (<F1>) {
	$o = $_;
	$o =~ s/\s+$//;
	$n = $o;
	$n =~ s/\.mrg/x\.mrg/;
	system("mv $o $n\n");
}

