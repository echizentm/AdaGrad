#!/usr/bin/perl
use strict;
use warnings;
use JSON::XS;
use AdaGradRDA;

my ($lambda) = @ARGV;

my $classifier = AdaGradRDA->new();
$classifier->lambda($lambda);

while (my $line = <STDIN>) {
    my $obj = decode_json($line);

    if ($obj->{label} == 0) {
        $obj->{label} = $classifier->classify(%$obj);
        print "classify: ".(encode_json($obj))."\n";
    } else {
        $classifier->update(%$obj);
        print "update: ".(encode_json($obj))."\n";
    }
}
print "wight: ".(encode_json($classifier->{weight}))."\n";

1;
