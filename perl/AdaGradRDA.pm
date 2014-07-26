package AdaGradRDA;
use strict;
use warnings;
use parent qw/Class::Accessor::Fast/;
use constant {
    DEFAULT_LAMBDA => 0.0,

    POSITIVE_LABEL =>  1,
    NEGATIVE_LABEL => -1,
    MARGIN         =>  1,
};

__PACKAGE__->mk_accessors(qw/lambda/);

sub new {
    my ($class) = @_;
    return $class->SUPER::new({
        lambda                   => DEFAULT_LAMBDA,
        weight                   => {},
        sum_of_gradients         => {},
        sum_of_squared_gradients => {},
        num_of_gradients         => 0,
    });
}

sub classify {
    my ($self, %args) = @_;
    return unless (__is_valid_data($args{data}));

    my $margin = 0.0;
    for my $feature (keys %{$args{data}}) {
        next unless ($self->{weight}{$feature});
        $margin += $self->{weight}{$feature} * $args{data}{$feature};
    }
    return $margin if ($args{as_margin});
    return ($margin > 0.0) ? POSITIVE_LABEL : NEGATIVE_LABEL;
}

sub update {
    my ($self, %args) = @_;
    return unless (__is_valid_label($args{label}));
    return unless (__is_valid_data($args{data}));

    return 1 if (($args{label} *
                  $self->classify(%args, as_margin => 1)
                 ) >= MARGIN);

    $self->{num_of_gradients}++;
    for my $feature (keys %{$args{data}}) {
        next if ($args{data}{$feature} == 0.0);
        my $gradient = -1 * $args{label} * $args{data}{$feature};
        $self->{sum_of_gradients}{$feature}         += $gradient;
        $self->{sum_of_squared_gradients}{$feature} += $gradient * $gradient;

        my $sign_of_gradients = ($self->{sum_of_gradients}{$feature} >= 0.0) ? 1 : -1;
        my $mean_of_gradients = ($sign_of_gradients *
                                 $self->{sum_of_gradients}{$feature} /
                                 $self->{num_of_gradients}
                                ) - $self->lambda;

        if ($mean_of_gradients <= 0.0) {
            delete $self->{weight}{$feature};
        } else {
            $self->{weight}{$feature} = -1 *
                                        $sign_of_gradients *
                                        $self->{num_of_gradients} *
                                        $mean_of_gradients /
                                        sqrt($self->{sum_of_squared_gradients}{$feature});
        }
    }
    return 1
}

sub __is_valid_label {
    my ($label) = @_;

    return unless ($label);
    return (($label == POSITIVE_LABEL) or
            ($label == NEGATIVE_LABEL)
           ) ? 1 : 0;
}

sub __is_valid_data {
    my ($data) = @_;

    return unless ($data);
    return (ref($data) eq 'HASH') ? 1 : 0;
}

1;
